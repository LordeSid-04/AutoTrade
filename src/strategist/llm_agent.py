"""
LLM Strategist (RAG + Structured Output)

The "brain" of the hierarchical system. Uses:
1. ChromaDB vector store for RAG retrieval of macro context
2. OpenAI GPT-4o for structural regime analysis
3. Pydantic schema for strict JSON output
4. Two-Stage Prompting: Reason first, extract second.
5. Self-Consistency: Poll N=3 times and aggregate via majority vote.
"""

import os
import logging
from collections import Counter
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Dict, List

load_dotenv()

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")

# 1. Define the Structured Output Schema
class StrategistConstraints(BaseModel):
    regime_classification: str = Field(description="One of: 'risk-on', 'risk-off', 'transitional'")
    confidence_score: int = Field(ge=1, le=10, description="Confidence in this regime call from 1 (lowest) to 10 (highest).")
    max_portfolio_volatility_target: float = Field(description="Max volatility constraint, e.g., 0.15")
    sector_exposure_caps: Dict[str, float] = Field(description="Dictionary of max weights per sector/asset, e.g., {'SPY': 0.40}")
    recommended_safe_haven: str = Field(description="The ticker of the best safe haven asset for the current regime (e.g., 'GLD', 'TLT', or 'CASH').")
    reasoning: str = Field(description="A concise paragraph explaining the rationale behind these constraints.")

class LLMStrategist:
    def __init__(self):
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        needs_ingestion = False
        if not os.path.exists(CHROMA_DB_DIR) or len(os.listdir(CHROMA_DB_DIR)) == 0:
            os.makedirs(CHROMA_DB_DIR, exist_ok=True)
            needs_ingestion = True

        self.vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=self.embedding_function
        )
        logger.info("LLMStrategist initialized with ChromaDB at %s", CHROMA_DB_DIR)
        
        if needs_ingestion:
            macro_dir = os.path.join(DATA_DIR, "macro_corpus")
            if os.path.exists(macro_dir):
                paths = [os.path.join(macro_dir, f) for f in os.listdir(macro_dir) if f.endswith(".txt")]
                if paths:
                    self.ingest_documents(paths)

    def ingest_documents(self, doc_paths):
        logger.info("Ingesting %d documents into Chroma...", len(doc_paths))
        docs = []
        for path in doc_paths:
            if os.path.exists(path):
                loader = TextLoader(path)
                loaded = loader.load()
                for doc in loaded:
                    doc_date_int = 19000101
                    if 'Date:' in doc.page_content:
                        try:
                            date_str = doc.page_content.split('Date:')[1].strip().split('\n')[0].strip()
                            doc_date_int = int(date_str.replace('-', ''))
                        except: pass
                    doc.metadata['date'] = doc_date_int
                docs.extend(loaded)

        if not docs:
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        self.vectorstore.add_documents(splits)
        logger.info("Successfully ingested %d chunks.", len(splits))

    def _dynamic_cap_scaling(self, caps: Dict[str, float], confidence: float) -> Dict[str, float]:
        """
        Relaxes the strictness of sector caps if the LLM confidence is low.
        Confidence 10 -> caps applied exactly as generated.
        Confidence 1 -> caps pushed towards 1.0 (uncapped).
        """
        scaled = {}
        # Relaxation factor: 0.0 at conf=10, 1.0 at conf=1
        relaxation = max(0.0, min(1.0, (10.0 - confidence) / 9.0))
        for ticker, cap in caps.items():
            # Interpolate between the raw cap and 1.0
            new_cap = cap + (1.0 - cap) * relaxation
            scaled[ticker] = round(new_cap, 4)
        return scaled

    def get_weekly_constraints(self, current_date_str, n_samples=3) -> tuple[StrategistConstraints, list]:
        """
        Retrieves context from RAG, calls LLM (N times for self-consistency),
        uses Two-Stage reasoning, scales caps by confidence, and returns JSON.
        """
        logger.info("Generating constraints for %s...", current_date_str)

        # 1. RAG Retrieval
        try:
            current_date_int = int(current_date_str.replace('-', ''))
            results = self.vectorstore.get(
                where={"date": {"$lte": current_date_int}},
                include=["metadatas", "documents"]
            )
            from langchain_core.documents import Document
            docs_with_dates = []
            if results and results['documents']:
                for i in range(len(results['documents'])):
                    doc_date = results['metadatas'][i].get("date", 19000101)
                    docs_with_dates.append((doc_date, Document(page_content=results['documents'][i], metadata=results['metadatas'][i])))
            docs_with_dates.sort(key=lambda x: x[0], reverse=True)
            filtered_docs = [doc for _, doc in docs_with_dates[:5]]
        except Exception as e:
            logger.error("Error retrieving from ChromaDB: %s", e)
            filtered_docs = []
        
        context = "\n".join([doc.page_content for doc in filtered_docs]) if filtered_docs else "No contemporaneous macro context available."

        # Inject Numeric Macro
        try:
            import yfinance as yf
            import pandas as pd
            end_date_pd = pd.to_datetime(current_date_str)
            start_date_pd = end_date_pd - pd.Timedelta(days=14)
            tickers = ["^VIX", "^TNX", "^IRX", "TIP", "IEF"]
            data = yf.download(tickers, start=start_date_pd.strftime('%Y-%m-%d'), end=(end_date_pd + pd.Timedelta(days=1)).strftime('%Y-%m-%d'), progress=False)['Close']
            
            if not data.empty:
                macro_indicators = f"\n\nNumeric Macro Indicators as of {current_date_str}:\n"
                if '^VIX' in data.columns: macro_indicators += f"- VIX: {data['^VIX'].dropna().iloc[-1]:.2f}\n"
                if '^TNX' in data.columns and '^IRX' in data.columns:
                    t10y, t3m = data['^TNX'].dropna().iloc[-1], data['^IRX'].dropna().iloc[-1]
                    macro_indicators += f"- 10Y Yield: {t10y:.2f}% | 3M Yield: {t3m:.2f}% | Spread: {t10y-t3m:.2f}%\n"
                context += macro_indicators
        except Exception: pass

        # Fallback Mock logic
        if not os.environ.get("OPENAI_API_KEY"):
            return StrategistConstraints(
                regime_classification="risk-on", confidence_score=5, max_portfolio_volatility_target=0.15,
                sector_exposure_caps={"SPY": 0.3}, recommended_safe_haven="CASH", reasoning="Mock"
            ), [{"page_content": context, "metadata": {}}]

        # 2. Setup LLMs (temperature 0.4 for diverse self-consistency)
        reasoning_llm = ChatOpenAI(model="gpt-4o", temperature=0.4)
        extraction_llm = ChatOpenAI(model="gpt-4o", temperature=0.0).with_structured_output(StrategistConstraints, method="function_calling")

        stage1_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a quantitative macro strategist. Analyze the provided macro context "
                       "and debate the current market regime. Consider inflation, yields, and policy statements. "
                       "Determine the regime ('risk-on', 'risk-off', 'transitional'), your confidence (1-10), "
                       "and how tightly sector exposures should be capped. Write a detailed free-text analysis."),
            ("human", "Current Date: {date}\nContext:\n{context}\n\nAnalysis:")
        ])

        stage2_prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract the structured portfolio constraints strictly from the provided macro analysis. "
                       "Rules: \n"
                       "1. regime_classification MUST be one of: 'risk-on', 'risk-off', 'transitional'.\n"
                       "2. If 'risk-off', leave safe havens ('TLT', 'GLD') uncapped (1.0).\n"
                       "Available Universe: SPY, QQQ, IWM, XLF, XLK, XLV, XLE, XLI, XLY, XLP, XLU, XLB, XLC, GLD, TLT."),
            ("human", "Analysis Document:\n{analysis}\n\nExtract constraints:")
        ])

        # Self-Consistency Loop
        samples = []
        for i in range(n_samples):
            try:
                logger.info(f"LLM Sampling {i+1}/{n_samples}...")
                analysis_output = (stage1_prompt | reasoning_llm).invoke({"date": current_date_str, "context": context}).content
                structured_output = (stage2_prompt | extraction_llm).invoke({"analysis": analysis_output})
                samples.append(structured_output)
            except Exception as e:
                logger.error(f"Sample {i+1} failed: {e}")

        if not samples:
            raise RuntimeError("All LLM samples failed.")

        # Aggregate Results (Majority Vote & Averages)
        regime_counts = Counter([s.regime_classification for s in samples])
        haven_counts = Counter([s.recommended_safe_haven for s in samples])
        
        maj_regime = regime_counts.most_common(1)[0][0]
        maj_haven = haven_counts.most_common(1)[0][0]
        avg_conf = np.mean([s.confidence_score for s in samples])
        avg_vol = np.mean([s.max_portfolio_volatility_target for s in samples])
        
        # Average the sector caps across samples
        all_tickers = set(k for s in samples for k in s.sector_exposure_caps.keys())
        avg_caps = {}
        for ticker in all_tickers:
            vals = [s.sector_exposure_caps.get(ticker, 1.0) for s in samples]
            avg_caps[ticker] = round(np.mean(vals), 4)

        # Apply Dynamic Scaling based on confidence
        final_caps = self._dynamic_cap_scaling(avg_caps, avg_conf)

        final_constraints = StrategistConstraints(
            regime_classification=maj_regime,
            confidence_score=int(round(avg_conf)),
            max_portfolio_volatility_target=round(avg_vol, 3),
            sector_exposure_caps=final_caps,
            recommended_safe_haven=maj_haven,
            reasoning=f"Self-consistency aggregated from {n_samples} samples. " + samples[0].reasoning
        )

        logger.info(f"Aggregated Constraints: Regime={maj_regime}, Conf={avg_conf:.1f}, Vol={avg_vol:.2f}")
        evidence = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in filtered_docs]
        return final_constraints, evidence

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    strategist = LLMStrategist()
    constraints, _ = strategist.get_weekly_constraints("2022-09-21", n_samples=2)
    print("\nGenerated Constraints (JSON):")
    print(constraints.model_dump_json(indent=2))
