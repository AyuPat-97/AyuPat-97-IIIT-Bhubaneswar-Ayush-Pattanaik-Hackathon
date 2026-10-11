import pandas as pd
from newsapi import NewsApiClient
from transformers import pipeline
import json
import re
import copy
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os


#  SAMPLE / SYNTHETIC DATASETS (Loads from data/ folder with fallback)                   

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
if not os.path.exists(_DATA_DIR):
    _DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def _load_json_data(filename):
    path = os.path.join(_DATA_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Warning] Failed to load {filename}: {e}")
    return []

MOCK_TWEETS = _load_json_data("sample_tweets.json")
MOCK_NEWS_HEADLINES = _load_json_data("sample_news.json")


# __________________________________________________________________________________________________

#  SECTOR KNOWLEDGE BASE

SECTOR_KNOWLEDGE_BASE = [
    {
        "theme": "oil_supply_disruption",
        "triggers": ["hormuz", "strait of hormuz", "opec cut", "opec production cut",
                     "oil embargo", "oil sanctions", "oil supply disruption", "oil supply shock",
                     "oil production cut", "oil blockade", "crude oil supply", "persian gulf",
                     "red sea tanker", "tanker seized", "crude supply bottleneck"],
        "impacts": {"XOM": 1},
        "direction_sensitive": True,
    },
    {
        "theme": "fuel_price_pressure",
        "triggers": ["fuel prices", "fuel price", "petrol prices", "gasoline prices",
                     "energy prices surging", "oil prices rising", "oil prices surge",
                     "energy costs rising", "crude prices", "brent crude",
                     "curb fuel prices", "mounting pressure to curb fuel",
                     "fuel prices remain", "painfully high", "painful fuel",
                     "fuel costs", "high fuel prices"],
        "impacts": {"XOM": 1},
        "direction_sensitive": False,
    },
    {
        "theme": "russia_energy_deal",
        "triggers": ["russia deal", "russia oil", "russia gas", "russia energy",
                     "russia ukraine ceasefire", "russia sanction", "russia fuel",
                     "trump russia deal", "shock russia deal"],
        "impacts": {"XOM": 1},
        "direction_sensitive": True,
    },
    {
        "theme": "rate_hike",
        "triggers": ["rate hike", "interest rate hike", "fed hike", "fed raises rates",
                     "rate increase", "hawkish fed", "tightening monetary policy",
                     "federal reserve hike"],
        "impacts": {"JPM": 1, "BAC": 1, "AAPL": -1, "MSFT": -1,
                    "GOOGL": -1, "AMZN": -1, "NVDA": -1, "TSLA": -1,
                    "META": -1, "NFLX": -1},
        "direction_sensitive": False,
    },
    {
        "theme": "rate_cut",
        "triggers": ["rate cut", "interest rate cut", "fed cut", "fed lowers rates",
                     "dovish fed", "easing monetary policy", "quantitative easing",
                     "rate reduction", "fed pivot"],
        "impacts": {"JPM": -1, "BAC": -1, "AAPL": 1, "MSFT": 1,
                    "GOOGL": 1, "AMZN": 1, "NVDA": 1, "TSLA": 1,
                    "META": 1, "NFLX": 1},
        "direction_sensitive": False,
    },
    {
        "theme": "ai_bubble_risk",
        "triggers": ["ai bubble", "artificial intelligence bubble", "ai overvalued",
                     "ai correction", "ai hype bubble", "ai selloff", "ai stocks drop",
                     "ai inflection point", "market correction ai",
                     "ai boom is approaching a dangerous"],
        "impacts": {"NVDA": -1, "MSFT": -1, "GOOGL": -1, "META": -1, "AAPL": -1},
        "direction_sensitive": False,
    },
    {
        "theme": "ai_boom",
        "triggers": ["ai demand surge", "ai adoption", "artificial intelligence boom",
                     "data center demand", "gpu demand", "llm adoption",
                     "generative ai growth", "ai infrastructure"],
        "impacts": {"NVDA": 1, "MSFT": 1, "GOOGL": 1, "AMZN": 1},
        "direction_sensitive": False,
    },
    {
        "theme": "chip_shortage",
        "triggers": ["chip shortage", "semiconductor shortage", "gpu shortage",
                     "chip supply shortage", "wafer shortage"],
        "impacts": {"NVDA": 1},
        "direction_sensitive": True,
    },
    {
        "theme": "recession_risk",
        "triggers": ["recession", "economic downturn", "economic hurricane",
                     "gdp contraction", "stagflation", "depression risk",
                     "market could drop", "markets could drop", "economic slowdown",
                     "hard landing"],
        "impacts": {"AAPL": -1, "MSFT": -1, "GOOGL": -1, "AMZN": -1,
                    "NVDA": -1, "TSLA": -1, "META": -1, "NFLX": -1,
                    "JPM": -1, "BAC": -1, "WMT": 0.3, "JNJ": 0.3},
        "direction_sensitive": False,
    },
    {
        "theme": "trade_war",
        "triggers": ["trade war", "tariff", "tariffs", "trade tariff", "import duty",
                     "trade dispute", "protectionism", "trade restrictions"],
        "impacts": {"AAPL": -1, "TSLA": -1, "NVDA": -1, "AMZN": -1,
                    "WMT": -1, "XOM": 0.3},
        "direction_sensitive": False,
    },
    {
        "theme": "china_risk",
        "triggers": ["china ban", "china crackdown", "us china tension", "china sanction",
                     "china restriction", "china tech ban", "taiwan tension",
                     "taiwan strait", "chip export ban"],
        "impacts": {"AAPL": -1, "NVDA": -1, "TSLA": -1, "GOOGL": -1},
        "direction_sensitive": False,
    },
    {
        "theme": "banking_crisis",
        "triggers": ["bank failure", "banking crisis", "bank run", "svb",
                     "bank collapse", "bank bailout", "financial crisis", "banking stress"],
        "impacts": {"JPM": -1, "BAC": -1, "BRK-B": -1},
        "direction_sensitive": False,
    },
    {
        "theme": "drug_approval",
        "triggers": ["fda approval", "fda approved", "drug approval", "regulatory approval",
                     "clinical trial success", "phase 3 success", "new drug approved"],
        "impacts": {"PFE": 1, "JNJ": 1},
        "direction_sensitive": False,
    },
    {
        "theme": "pharma_lawsuit",
        "triggers": ["drug lawsuit", "pharma lawsuit", "opioid lawsuit", "talc lawsuit",
                     "product liability", "drug recall", "medication recall"],
        "impacts": {"PFE": -1, "JNJ": -1},
        "direction_sensitive": False,
    },
    {
        "theme": "streaming_competition",
        "triggers": ["streaming wars", "streaming competition", "cord cutting",
                     "streaming subscriber loss", "streaming churn"],
        "impacts": {"NFLX": -1, "META": -1},
        "direction_sensitive": False,
    },
    {
        "theme": "ev_demand",
        "triggers": ["ev demand", "electric vehicle demand", "ev adoption",
                     "ev sales surge", "electric car sales"],
        "impacts": {"TSLA": 1},
        "direction_sensitive": False,
    },
    {
        "theme": "ev_slowdown",
        "triggers": ["ev slowdown", "ev demand drops", "electric vehicle slowdown",
                     "ev sales fall", "ev inventory glut"],
        "impacts": {"TSLA": -1},
        "direction_sensitive": False,
    },
]

# Pre-compile triggers: list of (phrase, index), sorted longest-first
_COMPILED_TRIGGERS = sorted(
    [(phrase.lower(), i)
     for i, entry in enumerate(SECTOR_KNOWLEDGE_BASE)
     for phrase in entry["triggers"]],
    key=lambda x: -len(x[0]),
)


# __________________________________________________________________________________________________

# Flexible semantic regex patterns that handle discontinuous wording, inverted syntax
# and separated qualifiers

THEME_SEMANTIC_PATTERNS = {
    "ai_bubble_risk": [
        re.compile(r'\b(ai|artificial\s+intelligence|genai|llm|gpus?|tech\s+stocks?|tech\s+sector)\b.{0,60}\b(bubble|burst\w*|overvalued|froth\w*|hype|overheated|meltdown|correction|selloff)\b', re.I),
        re.compile(r'\b(bubble|burst\w*|overvalued|froth\w*|hype|overheated|meltdown)\b.{0,60}\b(ai|artificial\s+intelligence|genai|llm|gpus?|tech\s+stocks?)\b', re.I),
    ],
    "ai_boom": [
        re.compile(r'\b(ai|artificial\s+intelligence|genai|llm|gpus?|data\s+centers?)\b.{0,50}\b(boom|demand|surge|growth|adoption|expansion|record|dominance|revolution)\b', re.I),
        re.compile(r'\b(surge|boom|growth|adoption|demand)\b.{0,50}\b(in\s+ai|for\s+ai|artificial\s+intelligence|ai\s+infrastructure)\b', re.I),
    ],
    "rate_hike": [
        re.compile(r'\b(interest\s+rates?|fed|federal\s+reserve|borrowing\s+costs?|policy\s+rates?|central\s+banks?)\b.{0,40}\b(hike\w*|rising|rises|rise|increase\w*|higher|tightening|hawkish|elevated|raising)\b', re.I),
        re.compile(r'\b(rising|higher|increasing|tightening|elevated|hiking|raising)\b.{0,40}\b(rates?|borrowing\s+costs?|interest\s+rates?)\b', re.I),
        re.compile(r'\b(rate\s+hikes?|interest\s+rate\s+hikes?)\b', re.I),
    ],
    "rate_cut": [
        re.compile(r'\b(interest\s+rates?|rates?|fed|federal\s+reserve|borrowing\s+costs?|policy\s+rates?|central\s+banks?)\b.{0,40}\b(cut\w*|lowering|lower\w*|decrease\w*|falling|easing|dovish|pivot|reduction|slash\w*)\b', re.I),
        re.compile(r'\b(cutting|lowering|falling|easing|dropping|slashing)\b.{0,40}\b(rates?|borrowing\s+costs?|interest\s+rates?)\b', re.I),
        re.compile(r'\b(rate\s+cuts?|interest\s+rate\s+cuts?)\b', re.I),
    ],
    "oil_supply_disruption": [
        re.compile(r'\b(hormuz|strait|persian\s+gulf|red\s+sea|opec|oil|crude|petroleum|tanker\w*)\b.{0,50}\b(disrupt\w*|blockade\w*|clos\w+|seiz\w+|attack\w*|strike\w*|embargo\w*|shortage\w*|paralyz\w+|halt\w*|threat\w*)\b', re.I),
        re.compile(r'\b(disruption|blockade|embargo|shortage|closure|crisis|paralyz\w+)\b.{0,50}\b(oil|crude|petroleum|energy|fuel|tanker\w*|shipping|traffic)\b', re.I),
    ],
    "fuel_price_pressure": [
        re.compile(r'\b(fuel|gasoline|petrol|diesel|oil|crude|energy)\b.{0,30}\b(prices?|costs?)\b.{0,40}\b(high|painfully|surging|surge|rising|elevated|spiking|spike|expensive|climb\w*)\b', re.I),
        re.compile(r'\b(surging|rising|high|painfully\s+high|elevated)\b.{0,40}\b(fuel|gasoline|petrol|energy)\b.{0,20}\b(prices?|costs?)\b', re.I),
        re.compile(r'\b(curb|combat|pressure\s+to\s+lower|high)\b.{0,30}\b(fuel|energy|petrol|gasoline)\b.{0,20}\b(prices?|costs?)\b', re.I),
    ],
    "chip_shortage": [
        re.compile(r'\b(chips?|semiconductors?|gpus?|wafers?|foundry|foundries|tsmc)\b.{0,50}\b(shortage\w*|bottleneck\w*|crunch|delays?|scarcity|constraints?|sold\s+out|capacity\s+limits?)\b', re.I),
        re.compile(r'\b(shortage|bottleneck|scarcity|crunch|delays?)\b.{0,50}\b(chips?|semiconductors?|gpus?|wafers?)\b', re.I),
    ],
    "recession_risk": [
        re.compile(r'\b(economy|economic|gdp|growth|markets?)\b.{0,40}\b(recession|downturn|slowdown|contraction|contracting|crash|stagflation|hard\s+landing|hurricane|slump|drop\s+20%)\b', re.I),
        re.compile(r'\b(risk\s+of|fears\s+of|warning\s+of|hard\s+landing)\b.{0,40}\b(recession|economic\s+collapse|hard\s+landing|downturn|slowdown)\b', re.I),
    ],
    "trade_war": [
        re.compile(r'\b(trade|imports?|exports?|tariff|tariffs|duties|duty)\b.{0,50}\b(war|disputes?|tensions?|retaliat\w+|imposed|slapped|escalat\w+|protectionism)\b', re.I),
        re.compile(r'\b(tariffs?|trade\s+barriers?|import\s+duties?)\b.{0,40}\b(announced|slapped|hiked|imposed|expanded)\b', re.I),
    ],
    "china_risk": [
        re.compile(r'\b(china|chinese|beijing|taiwan)\b.{0,50}\b(ban\w*|crackdown|sanctions?|restrictions?|tensions?|military|conflict|export\s+curbs?)\b', re.I),
    ],
    "banking_crisis": [
        re.compile(r'\b(banks?|banking|lenders?|deposits?|credit)\b.{0,50}\b(crisis|fail\w+|collapse\w*|run|panic|contagion|bailout|stress|distress|downgrade|flight)\b', re.I),
        re.compile(r'\b(failure|collapse|run\s+on|deposit\s+flight)\b.{0,40}\b(banks?|regional\s+banks?|lenders?)\b', re.I),
    ],
    "drug_approval": [
        re.compile(r'\b(drugs?|medications?|treatments?|therap\w+|fda|clinical\s+trials?|phase\s+3|regulators?)\b.{0,50}\b(approv\w+|cleared|clearance|success\w*|positive|greenlight|authorized)\b', re.I),
        re.compile(r'\b(approv\w+|clears?|cleared|clearance|authoriz\w+)\b.{0,50}\b(drugs?|medications?|treatments?|therap\w+|vaccines?)\b', re.I),
    ],
    "pharma_lawsuit": [
        re.compile(r'\b(drugs?|pharma|talc|talcum|opioids?|medications?|vaccines?)\b.{0,50}\b(lawsuits?|sued|settlement|verdict|recall\w*|liabilit\w+|litigation|class\s+action)\b', re.I),
    ],
    "ev_demand": [
        re.compile(r'\b(evs?|electric\s+vehicles?|tesla)\b.{0,40}\b(demand|sales|deliveries|orders)\b.{0,40}\b(surge\w*|beat\w*|record|boom|jump\w*|growth|strong)\b', re.I),
        re.compile(r'\b(sales|deliveries|demand|orders)\b.{0,30}\b(for|of)\b.{0,30}\b(evs?|electric\s+vehicles?|tesla)\b.{0,30}\b(surge\w*|jump\w*|boom|record|strong|beat\w*)\b', re.I),
    ],
    "ev_slowdown": [
        re.compile(r'\b(evs?|electric\s+vehicles?)\b.{0,40}\b(demand|sales|deliveries|orders)\b.{0,40}\b(slowdown|slump|drop\w*|fall\w*|glut|weak\w*|piling|slows?\s+down)\b', re.I),
        re.compile(r'\b(demand|sales|deliveries|orders)\b.{0,30}\b(for|of)\b.{0,30}\b(evs?|electric\s+vehicles?)\b.{0,30}\b(slowdown|slump|drop\w*|fall\w*|glut|weak\w*|piling|slows?\s+down)\b', re.I),
    ],
    "streaming_competition": [
        re.compile(r'\b(streaming|netflix)\b.{0,50}\b(wars?|competition|churn|subscriber\s+loss|missed|cancellations?)\b', re.I),
    ],
}

# __________________________________________________________________________________________________

SUPPLY_RESOLUTION_SIGNALS = frozenset({
    "reopened", "reopen", "no longer", "resolved", "lifted", "resumed",
    "restored", "eased", "cleared", "open again", "agreement reached",
    "ceasefire", "peace", "normalised", "normalized", "deal reached",
})

SUPPLY_DISRUPTION_SIGNALS = frozenset({
    "closed", "closure", "blockade", "seized", "disrupted", "disruption",
    "shortage", "embargo", "sanction", "halt", "halted", "suspended",
    "restricted", "military", "attack", "conflict", "threat", "blocked",
    "strike", "pressure", "rising", "surge", "surging", "paralyzed", "paralysed",
    "paralyze", "paralyse", "painfully high", "painful", "elevated", "spiking",
    "spike", "crisis", "chokepoint", "bottleneck", "high", "remain high",
    "remains high", "painfully",
})

CONTRAST_SIGNALS = frozenset({
    "and yet", "yet", "still", "remains", "remain", "nevertheless",
    "despite", "although", "even though", "however", "nonetheless",
})

# __________________________________________________________________________________________________

# High-precision mapping from Sector Knowledge Base themes directly to canonical events

THEME_EVENT_MAP = {
    "rate_hike": ("Central Bank & Monetary Policy", 0.94),
    "rate_cut": ("Central Bank & Monetary Policy", 0.94),
    "oil_supply_disruption": ("Geopolitical Event", 0.92),
    "fuel_price_pressure": ("Macroeconomic Shift", 0.89),
    "russia_energy_deal": ("Geopolitical Event", 0.91),
    "recession_risk": ("Macroeconomic Shift", 0.93),
    "trade_war": ("Geopolitical Event", 0.94),
    "china_risk": ("Geopolitical Event", 0.90),
    "banking_crisis": ("Credit & Banking Stress", 0.96),
    "drug_approval": ("Regulatory & Legal Risk", 0.93),
    "pharma_lawsuit": ("Regulatory & Legal Risk", 0.93),
    "ai_boom": ("Product Launch & Innovation", 0.91),
    "ai_bubble_risk": ("Market Valuation & Bubble", 0.93),
    "chip_shortage": ("Macroeconomic Shift", 0.89),
    "ev_demand": ("Product Launch & Innovation", 0.88),
    "ev_slowdown": ("Macroeconomic Shift", 0.88),
    "streaming_competition": ("Market Valuation & Bubble", 0.87),
}

# Semantic anchor terms matching the 13 macro theme hypotheses
MARKET_ANCHOR_TERMS = frozenset({
    "ai", "artificial intelligence", "genai", "llm", "gpu", "gpus", "bubble", "overvalued",
    "datacenter", "data center", "semiconductor", "chip", "chips", "foundry", "wafer",
    "fed", "federal reserve", "interest rate", "borrowing cost", "rate hike", "rate cut",
    "monetary policy", "hawkish", "dovish", "tightening", "easing", "inflation",
    "oil", "crude", "energy", "fuel", "gasoline", "petrol", "brent", "opec", "hormuz", "tanker",
    "recession", "economic downturn", "hard landing", "stagflation", "contraction", "economic hurricane",
    "bank run", "banking crisis", "credit crisis", "bank failure", "contagion", "svb", "bailout",
    "fda", "clinical trial", "drug approval", "lawsuit", "drug recall", "talc", "opioid", "pharma",
    "ev", "evs", "electric vehicle", "deliveries", "streaming wars", "cord cutting",
    "tariff", "tariffs", "trade war", "import duty", "trade dispute", "china ban", "taiwan"
})

# __________________________________________________________________________________________________


# Reputable premier financial news outlets to prevent irrelevant entertainment/pop-culture news
REPUTABLE_FINANCIAL_DOMAINS = (
    "bloomberg.com,reuters.com,cnbc.com,wsj.com,marketwatch.com,"
    "ft.com,barrons.com,finance.yahoo.com,investing.com,thestreet.com,"
    "benzinga.com,seekingalpha.com,fortune.com,fool.com,economist.com,"
    "forbes.com,nasdaq.com,businessinsider.com,investors.com,"
    "morningstar.com,marketbeat.com"
)

FINANCIAL_RELEVANCE_TERMS = frozenset({
    "stock", "stocks", "shares", "market", "markets", "earnings", "revenue", "fed",
    "rate", "rates", "inflation", "economy", "economic", "invest", "investor", "investors",
    "quarter", "quarterly", "profit", "profits", "margin", "margins", "nasdaq", "s&p",
    "dow", "dividend", "yield", "oil", "crude", "energy", "fuel", "tariff", "tariffs",
    "trade", "debt", "bond", "bonds", "treasury", "bank", "banking", "credit", "recession",
    "valuation", "fund", "etf", "ai", "tech", "chip", "chips", "semiconductor", "guidance",
    "sales", "deliveries", "antitrust", "sec", "fda", "merger", "acquisition", "buyout",
    "wall street", "ipo", "equity", "equities", "central bank", "interest rate", "capital",
    "analyst", "downgrade", "upgrade", "price target", "bull", "bear", "rally", "selloff",
    "portfolio", "hedge fund", "brokerage", "regulator", "monetary", "treasuries",
})

# __________________________________________________________________________________________________


# News query targeted exclusively to our 15 portfolio stocks and authoritative macro institutions
PORTFOLIO_NEWS_QUERY = (
    '("Apple" OR "AAPL" OR "Microsoft" OR "MSFT" OR "Google" OR "Alphabet" OR '
    '"Amazon" OR "AMZN" OR "Nvidia" OR "NVDA" OR "Meta Platforms" OR "Tesla" OR "TSLA" OR '
    '"JPMorgan" OR "Johnson & Johnson" OR "ExxonMobil" OR "Exxon" OR "Walmart" OR "Berkshire Hathaway" OR '
    '"Pfizer" OR "Netflix" OR "Bank of America" OR '
    '"Federal Reserve" OR "FOMC" OR "Jerome Powell" OR "interest rates" OR "OPEC" OR "crude oil" OR "tariffs")'
)

# Negation and dismissal phrases indicating risk fears are dismissed, rebutted, or overblown
DISMISSAL_SIGNALS = frozenset({
    "shrugged off", "shrugs off", "shrugged", "dismissed", "dismisses",
    "downplayed", "downplays", "played down", "plays down", "rejected",
    "rejects", "denied", "denies", "no sign of", "rebuffed", "disputed",
    "unlikely", "not worried", "rules out", "doubtful", "dispelled",
    "no evidence of", "overblown", "not a bubble", "not in a bubble",
    "no recession", "denies rumor", "pours cold water", "calms fears",
})

# Required context keywords for tech-specific macro themes (prevents unrelated auto/retail chatter from triggering tech themes)
TECH_CONTEXT_TERMS = frozenset({
    "tech", "technology", "software", "semiconductor", "chip", "chips",
    "gpu", "cloud", "data center", "datacenter", "nasdaq", "wall street",
    "silicon valley", "nvidia", "microsoft", "google", "alphabet", "meta",
    "apple", "amazon", "amd", "intel", "tsmc", "ai stocks", "tech stocks",
    "hyperscaler", "hyperscalers", "server", "servers",
})


# __________________________________________________________________________________________________


class RiskEngine:
    """
    The core AI/NLP Risk Engine with a semantic Sector Knowledge Base.

    Architecture
    ------------
    1. Data Ingestion  -- NewsAPI + simulated social media feed
    2. Direct Entity Recognition (Passes 1-3): cashtags, NER model, keyword scan
    3. Thematic Understanding (Pass 4): SECTOR_KNOWLEDGE_BASE maps macro events
       to affected portfolio tickers WITH directional impact.
    4. NLP pipelines: FinBERT + VADER sentiment, BART zero-shot event classification
    5. Per-ticker sentiment blending: NLP score blended with domain-knowledge override
    """

    PORTFOLIO = {
        "AAPL":  {"name": "Apple",          "weight": 1/15},
        "MSFT":  {"name": "Microsoft",      "weight": 1/15},
        "GOOGL": {"name": "Alphabet",       "weight": 1/15},
        "AMZN":  {"name": "Amazon",         "weight": 1/15},
        "NVDA":  {"name": "NVIDIA",         "weight": 1/15},
        "META":  {"name": "Meta",           "weight": 1/15},
        "TSLA":  {"name": "Tesla",          "weight": 1/15},
        "JPM":   {"name": "JPMorgan",       "weight": 1/15},
        "JNJ":   {"name": "Johnson & J",    "weight": 1/15},
        "XOM":   {"name": "Exxon Mobil",    "weight": 1/15},
        "WMT":   {"name": "Walmart",        "weight": 1/15},
        "BRK-B": {"name": "Berkshire",      "weight": 1/15},
        "PFE":   {"name": "Pfizer",         "weight": 1/15},
        "NFLX":  {"name": "Netflix",        "weight": 1/15},
        "BAC":   {"name": "Bank of America", "weight": 1/15},
    }

    PORTFOLIO_MAP = {
        "apple": "AAPL", "iphone": "AAPL", "mac": "AAPL", "ios": "AAPL",
        "microsoft": "MSFT", "azure": "MSFT", "windows": "MSFT", "copilot": "MSFT",
        "google": "GOOGL", "alphabet": "GOOGL", "gemini": "GOOGL", "youtube": "GOOGL",
        "amazon": "AMZN", "aws": "AMZN",
        "nvidia": "NVDA", "h100": "NVDA", "blackwell": "NVDA",
        "meta": "META", "facebook": "META", "instagram": "META", "whatsapp": "META",
        "tesla": "TSLA", "elon musk": "TSLA",
        "jpmorgan": "JPM", "jp morgan": "JPM", "jamie dimon": "JPM",
        "johnson & johnson": "JNJ", "j&j": "JNJ",
        "exxon": "XOM", "exxon mobil": "XOM", "exxonmobil": "XOM",
        "walmart": "WMT",
        "berkshire hathaway": "BRK-B", "berkshire": "BRK-B",
        "warren buffett": "BRK-B", "buffett": "BRK-B",
        "pfizer": "PFE",
        "netflix": "NFLX",
        "bank of america": "BAC", "bofa": "BAC",
    }

    NER_MIN_LEN = 3
    NER_TRUSTED_TYPES = {"ORG", "PER"}
    CASHTAG_RE = re.compile(r'\$([A-Z]{1,5}(?:-[A-Z])?)')

    FINANCIAL_SLANG = {
        "to the moon":      +0.55,
        "moon shot":        +0.45,
        "bull run":         +0.40,
        "buy the dip":      +0.35,
        "massive beat":     +0.40,
        "crushed it":       +0.40,
        "on fire":          +0.30,
        "going long":       +0.35,
        "long forever":     +0.40,
        "long position":    +0.30,
        "looking bullish":  +0.45,
        "fda approval":     +0.50,
        "huge catalyst":    +0.40,
        "surging":          +0.25,
        "beat vs":          +0.30,
        "locked in":        +0.25,
        "just dropped the": +0.30,
        "sell the rip":     -0.55,
        "sell the rally":   -0.45,
        "gonna bleed":      -0.50,
        "gonna hurt":       -0.40,
        "gonna feel it":    -0.35,
        "shorting hard":    -0.55,
        "short opportunity":-0.40,
        "downgraded":       -0.45,
        "missed badly":     -0.50,
        "missed estimates": -0.40,
        "risk-off signal":  -0.40,
        "risk event":       -0.35,
        "new lawsuit":      -0.40,
        "bubble concerns":  -0.45,
        "economic hurricane": -0.55,
        "markets could drop": -0.50,
        "could drop":       -0.35,
    }

    SEMANTIC_THEME_HYPOTHESES = {
        "artificial intelligence bubble, tech overvaluation, and AI market selloff risk": {
            "tickers": ["NVDA", "MSFT", "GOOGL", "META", "AAPL"],
            "override": -1.0,
            "theme": "AI Bubble & Tech Valuation Risk",
        },
        "artificial intelligence boom, surge in AI demand, and GPU infrastructure growth": {
            "tickers": ["NVDA", "MSFT", "GOOGL", "AMZN"],
            "override": 1.0,
            "theme": "AI Growth & Infrastructure Boom",
        },
        "interest rate hikes, rising borrowing costs, and hawkish monetary tightening": {
            "tickers": ["JPM", "BAC", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX"],
            "custom_overrides": {
                "JPM": 1.0, "BAC": 1.0,
                "AAPL": -1.0, "MSFT": -1.0, "GOOGL": -1.0, "AMZN": -1.0,
                "NVDA": -1.0, "TSLA": -1.0, "META": -1.0, "NFLX": -1.0,
            },
            "theme": "Rate Hike & Monetary Tightening",
        },
        "interest rate cuts, falling borrowing costs, and dovish monetary easing": {
            "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX", "JPM", "BAC"],
            "custom_overrides": {
                "JPM": -1.0, "BAC": -1.0,
                "AAPL": 1.0, "MSFT": 1.0, "GOOGL": 1.0, "AMZN": 1.0,
                "NVDA": 1.0, "TSLA": 1.0, "META": 1.0, "NFLX": 1.0,
            },
            "theme": "Rate Cut & Monetary Easing",
        },
        "crude oil supply disruption, geopolitical chokepoint conflict, or high fuel prices": {
            "tickers": ["XOM"],
            "override": 1.0,
            "theme": "Energy & Fuel Price Pressure",
        },
        "economic recession, macro contraction, and hard landing downturn": {
            "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX", "JPM", "BAC"],
            "override": -1.0,
            "theme": "Macroeconomic Recession Risk",
        },
        "semiconductor chip shortage, foundry delay, and supply chain bottlenecks": {
            "tickers": ["NVDA"],
            "override": 1.0,
            "theme": "Semiconductor Supply Constraints",
        },
        "banking crisis, bank run, and financial credit contagion": {
            "tickers": ["JPM", "BAC", "BRK-B"],
            "override": -1.0,
            "theme": "Banking Crisis & Credit Contagion",
        },
        "pharmaceutical regulatory FDA approval or successful clinical trial": {
            "tickers": ["PFE", "JNJ"],
            "override": 1.0,
            "theme": "Pharma Regulatory Breakthrough",
        },
        "pharmaceutical product liability lawsuit, drug recall, or litigation risk": {
            "tickers": ["PFE", "JNJ"],
            "override": -1.0,
            "theme": "Pharma Litigation Risk",
        },
        "electric vehicle sales boom, robust deliveries, and strong consumer adoption": {
            "tickers": ["TSLA"],
            "override": 1.0,
            "theme": "EV Demand Expansion",
        },
        "electric vehicle demand slowdown, sales slump, and inventory glut": {
            "tickers": ["TSLA"],
            "override": -1.0,
            "theme": "EV Demand Contraction",
        },
        "international trade war, protective tariffs, and import duties": {
            "tickers": ["AAPL", "TSLA", "NVDA", "AMZN", "WMT"],
            "override": -1.0,
            "theme": "Trade War & Tariffs",
        },
    }

    def __init__(self, news_api_key):
        self.newsapi = NewsApiClient(api_key=news_api_key)
        self._analysis_cache = {}
        self._load_nlp_models()

    def _load_nlp_models(self):
        print("Loading NLP Models...")
        torch.set_num_threads(4)
        self.ner_pipeline = pipeline("ner", model="dslim/bert-base-NER",
                                     aggregation_strategy="simple")
        self.roberta_pipeline = pipeline("sentiment-analysis",
                                         model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.sentiment_pipeline = pipeline("sentiment-analysis",
                                           model="ProsusAI/finbert")
        self.vader = SentimentIntensityAnalyzer()
        self.event_pipeline = pipeline("zero-shot-classification",
                                       model="facebook/bart-large-mnli")
        self.event_categories = [
            "Market Valuation & Bubble", "Central Bank & Monetary Policy",
            "Geopolitical Event", "Macroeconomic Shift", "Credit & Banking Stress",
            "Merger & Acquisition", "Product Launch & Innovation",
            "Earnings Report", "Regulatory & Legal Risk",
        ]
        print("Models loaded.")

    # --------------------------------------------------------------------------
    #  Data Ingestion
    # --------------------------------------------------------------------------

    def fetch_news_articles(self, page_num=1, page_size=5):
        texts = []
        try:
            resp = self.newsapi.get_everything(
                q=PORTFOLIO_NEWS_QUERY,
                domains=REPUTABLE_FINANCIAL_DOMAINS,
                language="en",
                sort_by="publishedAt",
                page_size=min(page_size * 5, 30),
                page=page_num,
            )
            for art in resp.get("articles", []):
                title = (art.get("title") or "").strip()
                desc  = (art.get("description") or "").strip()
                combined = (title + ". " + desc).strip()
                if not combined or combined == ".":
                    continue
                # Strictly accept ONLY stories that directly link to our portfolio stocks or macro drivers
                if self.has_portfolio_impact(combined):
                    texts.append(combined)
                    if len(texts) >= page_size:
                        break
        except Exception as e:
            print(f"[NewsAPI Warning] {e}")

        # Ensure exactly page_size news articles are always provided (supplementing if NewsAPI returns fewer)
        if len(texts) < page_size:
            start_idx = ((page_num - 1) * page_size) % len(MOCK_NEWS_HEADLINES)
            for i in range(page_size - len(texts)):
                idx = (start_idx + i) % len(MOCK_NEWS_HEADLINES)
                texts.append(MOCK_NEWS_HEADLINES[idx])

        return texts[:page_size]

    def fetch_social_feed(self, page_num=1, page_size=5):
        if not MOCK_TWEETS:
            return []
        start_idx = ((page_num - 1) * page_size) % len(MOCK_TWEETS)
        end_idx = start_idx + page_size
        if end_idx <= len(MOCK_TWEETS):
            return MOCK_TWEETS[start_idx:end_idx]
        else:
            return MOCK_TWEETS[start_idx:] + MOCK_TWEETS[: end_idx % len(MOCK_TWEETS)]

    # --------------------------------------------------------------------------
    #  Entity Extraction
    # --------------------------------------------------------------------------

    def _safe_truncate(self, text, max_chars=400):
        return text[:max_chars]

    @staticmethod
    def _word_boundary_match(keyword, text):
        pattern = r'(?<![a-z])' + re.escape(keyword) + r'(?![a-z])'
        return bool(re.search(pattern, text))

    def _extract_direct_tickers(self, text):
        """Passes 1-3: cashtags, fast keyword boundary scan, and NER fallback."""
        tickers = set()

        # Pass 1 - Cashtags (regex)
        for match in self.CASHTAG_RE.findall(text.upper()):
            if match in self.PORTFOLIO:
                tickers.add(match)

        # Pass 2 - Fast Keyword scan (dictionary lookup)
        lower = text.lower()
        for keyword, ticker in self.PORTFOLIO_MAP.items():
            if self._word_boundary_match(keyword, lower):
                tickers.add(ticker)

        # Pass 3 - NER fallback (only run if cashtags and direct keywords found nothing)
        if not tickers:
            try:
                entities = self.ner_pipeline(self._safe_truncate(text, max_chars=250))
                for ent in entities:
                    if ent.get("entity_group", "") not in self.NER_TRUSTED_TYPES:
                        continue
                    if ent.get("score", 0) < 0.70:
                        continue
                    word = ent.get("word", "").lower().strip()
                    if len(word) < self.NER_MIN_LEN:
                        continue
                    for keyword, ticker in self.PORTFOLIO_MAP.items():
                        if self._word_boundary_match(keyword, word):
                            tickers.add(ticker)
                            break
            except Exception as e:
                print(f"[NER Warning] {e}")

        return tickers

    @staticmethod
    def _is_trigger_dismissed(phrase, text_lower):
        pos = text_lower.find(phrase)
        if pos == -1:
            return False
        window = text_lower[max(0, pos - 75): min(len(text_lower), pos + len(phrase) + 45)]
        return any(d in window for d in DISMISSAL_SIGNALS)

    def _extract_thematic_tickers(self, text):
        """
        Pass 4: Sector Knowledge Base scan.
        Returns (set of tickers, dict of {ticker: directional_override}, primary_theme).
        Handles contrastive clauses, negation/dismissal ("shrugged off concerns about"),
        and ensures tech-specific themes only attach to articles with tech market context.
        """
        lower = text.lower()
        tickers   = set()
        overrides = {}
        matched   = set()
        primary_theme = None

        has_contrast = any(c in lower for c in CONTRAST_SIGNALS)
        has_disruption = any(s in lower for s in SUPPLY_DISRUPTION_SIGNALS)

        if has_contrast and has_disruption:
            is_resolving = False
        else:
            n_disruption = sum(1 for s in SUPPLY_DISRUPTION_SIGNALS if s in lower)
            n_resolution = sum(1 for s in SUPPLY_RESOLUTION_SIGNALS if s in lower)
            is_resolving = (n_resolution > n_disruption) and (n_resolution > 0)

        # Step 4(A): Literal trigger scan
        for phrase, idx in _COMPILED_TRIGGERS:
            if phrase not in lower:
                continue
            if idx in matched:
                continue
            entry = SECTOR_KNOWLEDGE_BASE[idx]
            theme_name = entry["theme"]

            # Guard 1: Negation / dismissal check (e.g. "shrugged off concerns about an AI bubble")
            if self._is_trigger_dismissed(phrase, lower):
                continue

            # Guard 2: Tech sector relevance check (prevents unrelated auto/retail chatter from triggering tech themes)
            if theme_name in {"ai_bubble_risk", "ai_boom", "chip_shortage"}:
                if not any(t in lower for t in TECH_CONTEXT_TERMS):
                    continue

            matched.add(idx)
            if not primary_theme:
                primary_theme = theme_name

            dir_multiplier = -1.0 if (entry.get("direction_sensitive") and is_resolving) else 1.0

            for ticker, direction in entry["impacts"].items():
                if ticker not in self.PORTFOLIO:
                    continue
                tickers.add(ticker)
                overrides[ticker] = overrides.get(ticker, 0.0) + (direction * dir_multiplier)

        # Step 4(B): Semantic flexible regex pattern scan
        for idx, entry in enumerate(SECTOR_KNOWLEDGE_BASE):
            if idx in matched:
                continue
            theme_name = entry["theme"]

            # Guard 1: Tech sector relevance check
            if theme_name in {"ai_bubble_risk", "ai_boom", "chip_shortage"}:
                if not any(t in lower for t in TECH_CONTEXT_TERMS):
                    continue

            patterns = THEME_SEMANTIC_PATTERNS.get(theme_name, [])
            matched_pattern = None
            for pat in patterns:
                m = pat.search(text)
                if m:
                    matched_pattern = m.group(0)
                    break
            if not matched_pattern:
                continue

            # Guard 2: Negation / dismissal check
            if self._is_trigger_dismissed(matched_pattern.lower(), lower):
                continue

            matched.add(idx)
            if not primary_theme:
                primary_theme = theme_name

            dir_multiplier = -1.0 if (entry.get("direction_sensitive") and is_resolving) else 1.0

            for ticker, direction in entry["impacts"].items():
                if ticker not in self.PORTFOLIO:
                    continue
                tickers.add(ticker)
                overrides[ticker] = overrides.get(ticker, 0.0) + (direction * dir_multiplier)

        return tickers, overrides, primary_theme

    def _extract_semantic_tickers(self, text):
        """
        Pass 5: Deep Learning Zero-Shot Theme Hypothesis Entailment.
        Uses BART-large-MNLI with independent multi-label hypothesis scoring (Sigmoid).
        Evaluates contextual macro narratives without rigid keyword dependencies.
        Returns (set of tickers, dict of {ticker: directional_override}).
        """
        lower = text.lower()
        # Fast pre-filter: if text has no market anchor terms, skip expensive 12-label BART evaluation
        if not any(term in lower for term in MARKET_ANCHOR_TERMS):
            return set(), {}

        candidate_labels = list(self.SEMANTIC_THEME_HYPOTHESES.keys())
        try:
            res = self.event_pipeline(
                self._safe_truncate(text, max_chars=250),
                candidate_labels=candidate_labels,
                multi_label=True
            )
            tickers = set()
            overrides = {}
            for label, score in zip(res["labels"], res["scores"]):
                if score >= 0.65:
                    info = self.SEMANTIC_THEME_HYPOTHESES[label]
                    theme_name = info.get("theme", "")

                    # Context Guard: tech-specific hypotheses require tech sector terms
                    if theme_name in {"AI Bubble & Tech Valuation Risk", "AI Growth & Infrastructure Boom", "Semiconductor Supply Constraints"}:
                        if not any(t in lower for t in TECH_CONTEXT_TERMS):
                            continue

                    # Dismissal Guard: do not trigger bubble risk if fears are dismissed
                    if theme_name == "AI Bubble & Tech Valuation Risk" and any(d in lower for d in DISMISSAL_SIGNALS):
                        continue

                    print(f"[Deep Learning Theme Match] '{theme_name}' (confidence: {score:.1%})")
                    if "custom_overrides" in info:
                        for t, ov in info["custom_overrides"].items():
                            if t in self.PORTFOLIO:
                                tickers.add(t)
                                overrides[t] = overrides.get(t, 0.0) + ov
                    else:
                        for t in info["tickers"]:
                            if t in self.PORTFOLIO:
                                tickers.add(t)
                                overrides[t] = overrides.get(t, 0.0) + info["override"]
            return tickers, overrides
        except Exception as e:
            print(f"[Semantic Theme Match Error] {e}")
            return set(), {}

    def extract_tickers(self, text):
        direct = self._extract_direct_tickers(text)
        thematic, _, _ = self._extract_thematic_tickers(text)
        tickers = (direct | thematic) & set(self.PORTFOLIO.keys())
        if not tickers:
            sem_tickers, _ = self._extract_semantic_tickers(text)
            tickers = set(sem_tickers) & set(self.PORTFOLIO.keys())
        return list(tickers)

    def has_portfolio_impact(self, text):
        direct = self._extract_direct_tickers(text)
        thematic, _, _ = self._extract_thematic_tickers(text)
        return bool((direct | thematic) & set(self.PORTFOLIO.keys()))

    # --------------------------------------------------------------------------
    #  NLP Sentiment
    # --------------------------------------------------------------------------

    def _get_sentiment_score(self, text, source):
        # 1. CardiffNLP Twitter-RoBERTa (State-of-the-art for social, tweets, and modern slang)
        try:
            rob_res = self.roberta_pipeline(self._safe_truncate(text, max_chars=200))[0]
            rob_label = rob_res["label"].lower()
            rob_conf = rob_res["score"]
            if rob_label == "positive":
                roberta_score = rob_conf
            elif rob_label == "negative":
                roberta_score = -rob_conf
            else:
                roberta_score = 0.0
        except Exception as e:
            print(f"[Twitter-RoBERTa Error] {e}")
            roberta_score = 0.0

        # 2. ProsusAI FinBERT (Specialized for formal corporate news & financial syntax)
        try:
            sent  = self.sentiment_pipeline(self._safe_truncate(text, max_chars=200))[0]
            label = sent["label"].lower()
            score = sent["score"]
            finbert_score = score if label == "positive" else (-score if label == "negative" else 0.0)
        except Exception as e:
            print(f"[FinBERT Error] {e}")
            finbert_score = 0.0

        # 3. Deep Learning Zero-Shot Bullish/Bearish Outlook (BART-large-MNLI)
        if (roberta_score >= 0.20 and finbert_score >= 0.20) or (roberta_score <= -0.20 and finbert_score <= -0.20):
            bart_score = (roberta_score + finbert_score) / 2.0
        else:
            try:
                bart_res = self.event_pipeline(
                    self._safe_truncate(text, max_chars=160),
                    candidate_labels=["bullish financial outlook", "bearish financial outlook", "neutral financial outlook"]
                )
                scores_dict = dict(zip(bart_res["labels"], bart_res["scores"]))
                bull_score = scores_dict.get("bullish financial outlook", 0.0)
                bear_score = scores_dict.get("bearish financial outlook", 0.0)
                bart_score = bull_score - bear_score
            except Exception as e:
                print(f"[BART Sentiment Error] {e}")
                bart_score = 0.0

        # 4. VADER auxiliary compound
        try:
            vader_compound = self.vader.polarity_scores(text)["compound"]
        except Exception as e:
            print(f"[VADER Error] {e}")
            vader_compound = 0.0

        # Determine context style (informal / social vs formal financial article)
        is_informal = (
            source == "Social"
            or bool(re.search(r'\$[A-Z]+|#\w+|!{1,}|stonks|moon|rip|gonna|bleed|beat|insane|drop', text, re.I))
        )

        if is_informal:
            # Social posts: Twitter-RoBERTa + VADER lexicon + Zero-Shot NLI dominate
            blended = 0.45 * roberta_score + 0.25 * bart_score + 0.15 * vader_compound + 0.15 * finbert_score
        else:
            # Formal posts: FinBERT + Zero-Shot NLI + VADER auxiliary
            blended = 0.45 * finbert_score + 0.25 * bart_score + 0.15 * roberta_score + 0.15 * vader_compound

        blended = self._apply_slang_correction(text, blended)

        if blended >= 0.12:
            label = "positive"
        elif blended <= -0.12:
            label = "negative"
        else:
            label = "neutral"

        return round(blended, 4), label

    def _apply_slang_correction(self, text, blended):
        lower = text.lower()
        delta = 0.0
        for phrase, d in self.FINANCIAL_SLANG.items():
            if phrase in lower:
                delta += d
        delta = max(-0.60, min(0.60, delta))
        corrected = blended + 0.35 * delta

        # Domain boundary protection for unmistakable bullish / bearish idioms:
        if any(p in lower for p in ["to the moon", "moon shot", "bull run", "going to the moon", "massive beat"]):
            corrected = max(0.45, corrected)
        elif any(p in lower for p in ["shorting hard", "gonna bleed", "sell the rip", "missed badly"]):
            corrected = min(-0.45, corrected)

        return max(-1.0, min(1.0, round(corrected, 4)))

    def _apply_thematic_override(self, base_score, overrides, ticker):
        # Blend base NLP score with domain-knowledge directional override.
        if ticker not in overrides:
            return base_score
        direction = overrides[ticker]
        domain_score = max(-0.85, min(0.85, direction * 0.50))

        # Polarity conflict guard:
        # If the NLP models have clear positive conviction (base_score >= 0.15) and
        # the domain override points in the opposite direction (domain_score < -0.10):
        # We must respect the text's explicit sentiment and NOT invert it into negative
        if (base_score >= 0.15 and domain_score < -0.10) or (base_score <= -0.15 and domain_score > 0.10):
            combined = 0.70 * base_score + 0.30 * domain_score
            if base_score > 0 and combined < 0:
                combined = 0.05
            elif base_score < 0 and combined > 0:
                combined = -0.05
            return max(-1.0, min(1.0, round(combined, 4)))

        combined = 0.50 * base_score + 0.50 * domain_score
        return max(-1.0, min(1.0, round(combined, 4)))

    # --------------------------------------------------------------------------
    #  Full Analysis
    # --------------------------------------------------------------------------

    def analyze_text(self, text, tickers, source="News", thematic_overrides=None, matched_theme=None):
        if thematic_overrides is None:
            thematic_overrides = {}

        cache_key = (text, tuple(sorted(tickers)), source, matched_theme, tuple(sorted(thematic_overrides.items())))
        if hasattr(self, "_analysis_cache") and cache_key in self._analysis_cache:
            return copy.deepcopy(self._analysis_cache[cache_key])

        base_sentiment, _ = self._get_sentiment_score(text, source)

        # Event Classification: authoritative domain mapping if theme was matched, else BART zero-shot
        if matched_theme and matched_theme in THEME_EVENT_MAP:
            primary_event, event_conf = THEME_EVENT_MAP[matched_theme]
        else:
            try:
                res = self.event_pipeline(self._safe_truncate(text, max_chars=200),
                                          candidate_labels=self.event_categories)
                primary_event = res["labels"][0]
                event_conf    = res["scores"][0]
            except Exception as e:
                print(f"[Event Error] {e}")
                primary_event, event_conf = "Macroeconomic", 0.5

        # Per-ticker sentiment (blended with domain override)
        ticker_sentiments = {
            t: self._apply_thematic_override(base_sentiment, thematic_overrides, t)
            for t in tickers
        }

        aggregate = round(sum(ticker_sentiments.values()) / len(ticker_sentiments), 3) \
                    if ticker_sentiments else base_sentiment

        if aggregate >= 0.12:
            sentiment_label = "positive"
        elif aggregate <= -0.12:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        impact_score = int(min(max(round((abs(aggregate) * 5) + (event_conf * 5)), 1), 10))

        if aggregate >= 0.40:
            trajectory = "Bullish"
        elif aggregate >= 0.12:
            trajectory = "Slightly Bullish"
        elif aggregate > -0.12:
            trajectory = "Neutral"
        elif aggregate > -0.40:
            trajectory = "Slightly Bearish"
        else:
            trajectory = "Bearish"

        result = {
            "source":               source,
            "text_snippet":         text[:120] + ("..." if len(text) > 120 else ""),
            "affected_tickers":     tickers,
            "ticker_sentiments":    ticker_sentiments,
            "sentiment_score":      aggregate,
            "sentiment_label":      sentiment_label,
            "event_classification": primary_event,
            "event_confidence":     round(event_conf, 3),
            "impact_score":         impact_score,
            "trajectory":           trajectory,
        }

        if hasattr(self, "_analysis_cache"):
            self._analysis_cache[cache_key] = copy.deepcopy(result)

        return result

    # --------------------------------------------------------------------------
    #  Public API
    # --------------------------------------------------------------------------

    def process_live_feed(self, page_num=1, news_limit=5, social_limit=5):
        all_signals = []
        seen_texts = set()

        with torch.inference_mode():
            print(f"Fetching live news (page {page_num}, limit {news_limit})...")
            for text in self.fetch_news_articles(page_num=page_num, page_size=news_limit):
                clean_text = text.strip()
                if clean_text and clean_text not in seen_texts:
                    seen_texts.add(clean_text)
                    direct = self._extract_direct_tickers(clean_text)
                    thematic, overrides, matched_theme = self._extract_thematic_tickers(clean_text)
                    tickers = set(direct | thematic) & set(self.PORTFOLIO.keys())
                    if not tickers:
                        sem_tickers, sem_overrides = self._extract_semantic_tickers(clean_text)
                        tickers = set(sem_tickers) & set(self.PORTFOLIO.keys())
                        overrides.update(sem_overrides)
                    if not tickers:
                        continue  # Scrap INDEX: strictly reject non-portfolio items
                    all_signals.append(
                        self.analyze_text(clean_text, list(tickers), "News", overrides, matched_theme=matched_theme))

            print(f"Processing social feed (page {page_num}, limit {social_limit})...")
            for text in self.fetch_social_feed(page_num=page_num, page_size=social_limit):
                clean_text = text.strip()
                if clean_text and clean_text not in seen_texts:
                    seen_texts.add(clean_text)
                    direct = self._extract_direct_tickers(clean_text)
                    thematic, overrides, matched_theme = self._extract_thematic_tickers(clean_text)
                    tickers = set(direct | thematic) & set(self.PORTFOLIO.keys())
                    if not tickers:
                        sem_tickers, sem_overrides = self._extract_semantic_tickers(clean_text)
                        tickers = set(sem_tickers) & set(self.PORTFOLIO.keys())
                        overrides.update(sem_overrides)
                    if not tickers:
                        continue  # Scrap INDEX: strictly reject non-portfolio items
                    all_signals.append(
                        self.analyze_text(clean_text, list(tickers), "Social", overrides, matched_theme=matched_theme))

        print(f"Generated {len(all_signals)} signals affecting portfolio stocks.")
        return all_signals

    def process_custom_text(self, text):
        with torch.inference_mode():
            direct   = self._extract_direct_tickers(text)
            thematic, overrides, matched_theme = self._extract_thematic_tickers(text)
            tickers  = set(direct | thematic) & set(self.PORTFOLIO.keys())

            # Pass 5: If no tickers found via direct or keyword KB, use Deep Learning Zero-Shot Semantic Matcher
            if not tickers:
                sem_tickers, sem_overrides = self._extract_semantic_tickers(text)
                tickers = set(sem_tickers) & set(self.PORTFOLIO.keys())
                overrides.update(sem_overrides)

            if not tickers:
                return None

            return self.analyze_text(text, list(tickers), "Manual Input", overrides, matched_theme=matched_theme)


if __name__ == "__main__":
    API_KEY = "46946b89199c424aa61b0ca70196d0f8"
    engine  = RiskEngine(news_api_key=API_KEY)
    print("\nScanning live market feed...")
    signals = engine.process_live_feed()
    print(json.dumps(signals, indent=2))
