import pandas as pd
import numpy as np
import os
import logging
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Ticker to PERMNO Mapping is not strictly needed since the user selected TICKER in the WRDS query,
# but we can use it to validate the universe.
TICKERS = ["SPY", "QQQ", "IWM", "XLF", "XLK", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLC", "GLD", "TLT"]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW_CSV_PATH = os.path.join(DATA_DIR, "wrds_crsp_raw.csv")

def preprocess_crsp_data():
    """
    Reads the raw CRSP daily stock file exported from the WRDS web interface,
    calculates institutional-grade features, and exports to processed_features.csv
    """
    if not os.path.exists(RAW_CSV_PATH):
        logger.error(f"Raw CRSP data not found at {RAW_CSV_PATH}. Please download it from WRDS and place it there.")
        return None

    logger.info(f"Loading raw CRSP data from {RAW_CSV_PATH}...")
    
    # Read CSV. Depending on the exact columns exported, we will map them.
    # We expect columns like: date, TICKER (or tic), PRC, VOL, RET, SHROUT, BIDLO, ASKHI, BID, ASK, OPENPRC, RETX, etc.
    # The header provided by the user was: PERMNO,date,TICKER,CUSIP,DLRET,BIDLO,ASKHI,PRC,VOL,RET,BID,ASK,SHROUT,OPENPRC,NUMTRD,RETX
    df = pd.read_csv(RAW_CSV_PATH, low_memory=False)
    
    # Standardize column names to upper case for easier mapping
    df.columns = [c.upper() for c in df.columns]
    
    logger.info(f"Loaded {len(df)} rows. Processing features...")
    
    # Parse dates
    # User's CSV shows DD/MM/YYYY (e.g. 19/06/2018). We must specify dayfirst=True
    df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True, errors='coerce')
    
    # Filter only our universe
    df = df[df['TICKER'].isin(TICKERS)].copy()
    
    # Coerce numeric columns, as CRSP uses 'C' or 'B' for missing returns sometimes
    for col in ['PRC', 'VOL', 'RET', 'BIDLO', 'ASKHI', 'BID', 'ASK', 'SHROUT', 'OPENPRC', 'RETX']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # CRSP Price can be negative if it's a bid/ask average. We take absolute value.
    df['PRC'] = df['PRC'].abs()
    
    processed_dfs = []
    
    for ticker, group in df.groupby('TICKER'):
        group = group.sort_values('DATE').copy()
        
        # We rename columns to match downstream RL expectations
        group.rename(columns={'DATE': 'Date', 'TICKER': 'Ticker', 'PRC': 'Close'}, inplace=True)
        
        # Handle returns. If RET is NaN, fill with 0
        group['RET'] = group['RET'].fillna(0.0)
        group['RETX'] = group['RETX'].fillna(0.0)
        
        # 1. Log Return (already adjusted for splits/divs in CRSP RET)
        group['Log_Return'] = np.log1p(group['RET'])
        
        # 2. Dividend Yield (RET - RETX)
        group['Div_Yield'] = group['RET'] - group['RETX']
        
        # 3. Overnight Return (Open / Prev Close - 1)
        # Forward fill missing opens with close just in case
        group['OPENPRC'] = group['OPENPRC'].fillna(group['Close'])
        group['Overnight_Ret'] = (group['OPENPRC'] / group['Close'].shift(1)) - 1
        group['Overnight_Ret'] = group['Overnight_Ret'].fillna(0.0)
        
        # 4. Parkinson Volatility (Intraday high/low volatility estimator)
        # Formula: sqrt( 1 / (4 * ln(2)) * (ln(High/Low))^2 )
        # If High/Low is missing, fallback to 0
        group['ASKHI'] = group['ASKHI'].abs()
        group['BIDLO'] = group['BIDLO'].abs()
        
        # Ensure High >= Low and strictly positive
        valid_hl = (group['ASKHI'] > 0) & (group['BIDLO'] > 0) & (group['ASKHI'] >= group['BIDLO'])
        high_low_ratio = np.where(valid_hl, group['ASKHI'] / group['BIDLO'], 1.0)
        
        park_var = (1.0 / (4.0 * np.log(2.0))) * (np.log(high_low_ratio) ** 2)
        # Annualized Parkinson Volatility
        group['Parkinson_Vol'] = np.sqrt(park_var) * np.sqrt(252)
        
        # 5. Bid-Ask Spread
        # (ASK - BID) / Close
        group['ASK'] = group['ASK'].abs()
        group['BID'] = group['BID'].abs()
        spread = (group['ASK'] - group['BID']) / group['Close']
        # Clip spread to reasonable bounds to prevent data errors (0 to 10%)
        group['BA_Spread'] = np.clip(spread.fillna(0.0), 0.0, 0.1)
        
        # 6. Dollar Volume (in millions)
        # CRSP Volume is usually in shares.
        group['Dollar_Volume'] = (group['Close'] * group['VOL'].fillna(0)) / 1e6
        
        # --- Traditional Features ---
        group['SMA_20'] = group['Close'].rolling(window=20).mean()
        group['SMA_50'] = group['Close'].rolling(window=50).mean()
        
        # RSI 14
        delta = group['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        group['RSI_14'] = 100 - (100 / (1 + rs))
        
        # Drop initial NaN rows caused by rolling windows
        group.dropna(subset=['SMA_50', 'RSI_14'], inplace=True)
        
        # Keep only the columns we need for the RL agent
        cols_to_keep = [
            'Date', 'Ticker', 'Close', 'Log_Return', 
            'SMA_20', 'SMA_50', 'Parkinson_Vol', 'RSI_14',
            'Div_Yield', 'Overnight_Ret', 'BA_Spread', 'Dollar_Volume'
        ]
        
        processed_dfs.append(group[cols_to_keep])
        
    final_df = pd.concat(processed_dfs, ignore_index=True)
    
    # Forward fill any remaining NaNs safely within each ticker group
    final_df = final_df.groupby('Ticker', group_keys=False).apply(lambda x: x.ffill().bfill())
    
    final_df.sort_values(by=['Date', 'Ticker'], inplace=True)
    
    # --- Dynamic Risk-Free Rate from FRED (DGS3MO) ---
    logger.info("Fetching dynamic risk-free rate from FRED DGS3MO (3-Month Treasury)...")
    date_min = final_df['Date'].min()
    date_max = final_df['Date'].max()
    try:
        import pandas_datareader.data as web
        # FRED rates are given in percent (e.g., 5.25 for 5.25%)
        rf_df = web.DataReader('DGS3MO', 'fred', start=date_min, end=date_max)
        if not rf_df.empty:
            rf_df = rf_df.reset_index()
            rf_df.columns = ['Date', 'Risk_Free_Rate']
            rf_df['Date'] = pd.to_datetime(rf_df['Date'])
            rf_df['Risk_Free_Rate'] = rf_df['Risk_Free_Rate'] / 100.0  # Convert from % to decimal
            
            final_df['Date'] = pd.to_datetime(final_df['Date'])
            final_df = final_df.merge(rf_df, on='Date', how='left')
            final_df['Risk_Free_Rate'] = final_df['Risk_Free_Rate'].ffill().bfill()
            logger.info("Dynamic FRED risk-free rate merged. Range: %.4f to %.4f",
                        final_df['Risk_Free_Rate'].min(), final_df['Risk_Free_Rate'].max())
        else:
            logger.warning("FRED DGS3MO returned empty. Falling back to 0.02 default.")
            final_df['Risk_Free_Rate'] = 0.02
    except Exception as e:
        logger.warning("Failed to fetch FRED DGS3MO: %s. Falling back to 0.02 default.", e)
        final_df['Risk_Free_Rate'] = 0.02
    
    # Reorder columns explicitly
    cols = ['Date', 'Ticker', 'Close', 'Log_Return', 'SMA_20', 'SMA_50', 'Parkinson_Vol', 'RSI_14', 'Div_Yield', 'Overnight_Ret', 'BA_Spread', 'Dollar_Volume', 'Risk_Free_Rate']
    final_df = final_df[cols]
    
    output_path = os.path.join(DATA_DIR, "processed_features.csv")
    final_df.to_csv(output_path, index=False)
    logger.info(f"Processed CRSP features successfully saved to {output_path}")
    logger.info(f"Total Rows: {len(final_df)}. Date Range: {final_df['Date'].min().date()} to {final_df['Date'].max().date()}")
    
    return final_df

if __name__ == "__main__":
    processed_df = preprocess_crsp_data()
    if processed_df is not None:
        logger.info("CRSP Data Pipeline Complete! You can now generate the expert dataset.")
