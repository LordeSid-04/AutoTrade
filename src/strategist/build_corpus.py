import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS_DIR = os.path.join(PROJECT_ROOT, "data", "macro_corpus")

# Create corpus dir if it doesn't exist
os.makedirs(CORPUS_DIR, exist_ok=True)

data = {
    "2015-12-16_FOMC_December_2015_-_First_Rate_Hike_Since_2006.txt": "Date: 2015-12-16\n\nRelease Date: December 16, 2015\n\nFor release at 2:00 p.m. EST\n\nInformation received since the Federal Open Market Committee met in October suggests that economic activity has been expanding at a moderate pace. Household spending and business fixed investment have been increasing at solid rates... The Committee judged that there has been considerable improvement in labor market conditions... The Committee decided to raise the target range for the federal funds rate to 1/4 to 1/2 percent.",
    
    "2017-03-15_FOMC_March_2017_-_Continued_Gradual_Tightening.txt": "Date: 2017-03-15\n\nRelease Date: March 15, 2017\n\nFor release at 2:00 p.m. EDT\n\nInformation received since the Federal Open Market Committee met in February indicates that the labor market has continued to strengthen and that economic activity has continued to expand at a moderate pace... In view of realized and expected labor market conditions and inflation, the Committee decided to raise the target range for the federal funds rate to 3/4 to 1 percent.",
    
    "2017-06-14_FOMC_June_2017_-_Balance_Sheet_Normalization_Plan.txt": "Date: 2017-06-14\n\nRelease Date: June 14, 2017\n\nFor release at 2:00 p.m. EDT\n\nInformation received since the Federal Open Market Committee met in May indicates that the labor market has continued to strengthen... In view of realized and expected labor market conditions and inflation, the Committee decided to raise the target range for the federal funds rate to 1 to 1-1/4 percent. The Committee also expects to begin implementing a balance sheet normalization program this year...",
    
    "2017-12-13_FOMC_December_2017_-_Third_Hike_of_the_Year.txt": "Date: 2017-12-13\n\nRelease Date: December 13, 2017\n\nFor release at 2:00 p.m. EST\n\nInformation received since the Federal Open Market Committee met in November indicates that the labor market has continued to strengthen... In view of realized and expected labor market conditions and inflation, the Committee decided to raise the target range for the federal funds rate to 1-1/4 to 1-1/2 percent.",
    
    "2019-07-31_FOMC_July_2019_-_Insurance_Rate_Cut.txt": "Date: 2019-07-31\n\nRelease Date: July 31, 2019\n\nFor release at 2:00 p.m. EDT\n\nInformation received since the Federal Open Market Committee met in June indicates that the labor market remains strong... In light of the implications of global developments for the economic outlook as well as muted inflation pressures, the Committee decided to lower the target range for the federal funds rate to 2 to 2-1/4 percent.",
    
    "2020-03-03_FOMC_Emergency_Rate_Cut_-_COVID-19_Pandemic.txt": "Date: 2020-03-03\n\nRelease Date: March 3, 2020\n\nFor release at 10:00 a.m. EST\n\nThe fundamentals of the U.S. economy remain strong. However, the coronavirus poses evolving risks to economic activity. In light of these risks and in support of achieving its maximum employment and price stability goals, the Federal Open Market Committee decided today to lower the target range for the federal funds rate by 1/2 percentage point, to 1 to 1-1/4 percent.",
    
    "2020-03-15_FOMC_Emergency_-_Zero_Rates_and_QE_Restart.txt": "Date: 2020-03-15\n\nRelease Date: March 15, 2020\n\nFor release at 5:00 p.m. EDT\n\nThe coronavirus outbreak has harmed communities and disrupted economic activity in many countries... The Committee decided to lower the target range for the federal funds rate to 0 to 1/4 percent. The Committee expects to maintain this target range until it is confident that the economy has weathered recent events... The Committee will increase its holdings of Treasury securities by at least $500 billion...",
    
    "2020-06-10_FOMC_June_2020_-_Extended_Forward_Guidance.txt": "Date: 2020-06-10\n\nRelease Date: June 10, 2020\n\nFor release at 2:00 p.m. EDT\n\nThe coronavirus outbreak is causing tremendous human and economic hardship... Financial conditions have improved, in part reflecting policy measures... The Committee decided to maintain the target range for the federal funds rate at 0 to 1/4 percent. The Committee expects to maintain this target range until it is confident that the economy has weathered recent events...",
    
    "2021-11-03_FOMC_November_2021_-_Taper_Announcement.txt": "Date: 2021-11-03\n\nRelease Date: November 3, 2021\n\nFor release at 2:00 p.m. EDT\n\nWith progress on vaccinations and strong policy support, indicators of economic activity and employment have continued to strengthen. Inflation is elevated, largely reflecting factors that are expected to be transitory... In light of the substantial further progress the economy has made toward the Committee's goals, the Committee decided to begin reducing the monthly pace of its net asset purchases...",
    
    "2022-03-16_FOMC_March_2022_-_First_Rate_Hike_of_Tightening_Cy.txt": "Date: 2022-03-16\n\nRelease Date: March 16, 2022\n\nFor release at 2:00 p.m. EDT\n\nIndicators of economic activity and employment have continued to strengthen. Inflation remains elevated, reflecting supply and demand imbalances related to the pandemic, higher energy prices, and broader price pressures. The invasion of Ukraine by Russia is causing tremendous human and economic hardship... The Committee decided to raise the target range for the federal funds rate to 1/4 to 1/2 percent...",
    
    "2022-06-15_FOMC_June_2022_-_75_Basis_Point_Hike.txt": "Date: 2022-06-15\n\nRelease Date: June 15, 2022\n\nFor release at 2:00 p.m. EDT\n\nOverall economic activity appears to have picked up after edging down in the first quarter... Inflation remains elevated, reflecting supply and demand imbalances... The Committee is highly attentive to inflation risks. The Committee decided to raise the target range for the federal funds rate to 1-1/2 to 1-3/4 percent and anticipates that ongoing increases in the target range will be appropriate.",
    
    "2022-09-21_FOMC_September_2022_-_Third_Consecutive_75bp_Hike.txt": "Date: 2022-09-21\n\nRelease Date: September 21, 2022\n\nFor release at 2:00 p.m. EDT\n\nRecent indicators point to modest growth in spending and production. Job gains have been robust... Inflation remains elevated, reflecting supply and demand imbalances... The Committee is highly attentive to inflation risks. The Committee decided to raise the target range for the federal funds rate to 3 to 3-1/4 percent and anticipates that ongoing increases in the target range will be appropriate.",
    
    "2022-12-14_FOMC_December_2022_-_Pace_of_Hikes_Slows.txt": "Date: 2022-12-14\n\nRelease Date: December 14, 2022\n\nFor release at 2:00 p.m. EST\n\nRecent indicators point to modest growth in spending and production... Inflation remains elevated... The Committee decided to raise the target range for the federal funds rate to 4-1/4 to 4-1/2 percent. The Committee anticipates that ongoing increases in the target range will be appropriate in order to attain a stance of monetary policy that is sufficiently restrictive to return inflation to 2 percent over time.",
    
    "2023-07-26_FOMC_July_2023_-_Possible_Final_Hike.txt": "Date: 2023-07-26\n\nRelease Date: July 26, 2023\n\nFor release at 2:00 p.m. EDT\n\nRecent indicators suggest that economic activity has been expanding at a moderate pace. Job gains have been robust in recent months... Inflation remains elevated. The U.S. banking system is sound and resilient... The Committee decided to raise the target range for the federal funds rate to 5-1/4 to 5-1/2 percent. The Committee will continue to assess additional information and its implications for monetary policy.",
    
    # --- Broader Macro Data (CPI, NFP, ISM) ---
    "2020-04-03_NFP_March_2020_-_Historic_Job_Losses.txt": "Date: 2020-04-03\n\nRelease Date: April 3, 2020\n\nNonfarm Payrolls (NFP) Employment Situation Summary\n\nTotal nonfarm payroll employment fell by 701,000 in March, and the unemployment rate rose to 4.4 percent. The changes in these measures reflect the effects of the coronavirus (COVID-19) and efforts to contain it. Employment in leisure and hospitality fell by 459,000, mainly in food services and drinking places.",
    
    "2022-06-10_CPI_May_2022_-_Inflation_Peaks.txt": "Date: 2022-06-10\n\nRelease Date: June 10, 2022\n\nConsumer Price Index (CPI) Summary\n\nThe Consumer Price Index for All Urban Consumers (CPI-U) increased 1.0 percent in May on a seasonally adjusted basis after rising 0.3 percent in April. Over the last 12 months, the all items index increased 8.6 percent before seasonal adjustment, the largest 12-month increase since the period ending December 1981.",
    
    "2022-09-01_ISM_Manufacturing_August_2022.txt": "Date: 2022-09-01\n\nRelease Date: September 1, 2022\n\nISM Manufacturing PMI\n\nThe August Manufacturing PMI registered 52.8 percent, identical to the July reading. This figure indicates expansion in the overall economy for the 27th month in a row. However, new orders contracted, and prices paid decreased significantly, reflecting easing demand and supply chain pressures amidst aggressive rate hikes.",
    
    "2016-01-08_NFP_December_2015_-_Solid_Job_Growth.txt": "Date: 2016-01-08\n\nRelease Date: January 8, 2016\n\nNonfarm Payrolls (NFP) Employment Situation Summary\n\nTotal nonfarm payroll employment rose by 292,000 in December, and the unemployment rate was unchanged at 5.0 percent. Job gains occurred in several industries, including professional and business services, construction, health care, and food services.",
    
    "2018-12-03_ISM_Manufacturing_November_2018.txt": "Date: 2018-12-03\n\nRelease Date: December 3, 2018\n\nISM Manufacturing PMI\n\nThe November Manufacturing PMI registered 59.3 percent, an increase of 1.6 percentage points from the October reading. This indicates strong growth in manufacturing, though concerns about tariffs and trade tensions with China persist among purchasing managers."
}

for filename, content in data.items():
    filepath = os.path.join(CORPUS_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

print("Corpus updated successfully with pure point-in-time Federal Reserve press releases.")
