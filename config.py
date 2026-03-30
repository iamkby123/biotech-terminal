import os
from dotenv import load_dotenv

load_dotenv()

CTGOV_BASE = "https://clinicaltrials.gov/api/v2/studies"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_COMPANY_TICKERS = "https://www.sec.gov/files/company_tickers.json"
EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
FDA_PDUFA_URL = "https://www.rttnews.com/corpinfo/fdacalendar.aspx"
FDA_ADCOM_URL = "https://www.fda.gov/advisory-committees/advisory-committee-calendar"
PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_SUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

DATABASE_URL = os.environ.get("DATABASE_URL", "")

NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "BiotechTerminal contact@example.com")

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "ct2qh1pr01qhb1amkn3gct2qh1pr01qhb1amkn40")
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
FINNHUB_RATE_LIMIT = 1.1  # seconds between requests (60 req/min free tier)

CTGOV_PAGE_SIZE = 100
CTGOV_RATE_LIMIT = 0.12
EDGAR_RATE_LIMIT = 0.12
PUBMED_RATE_LIMIT = 0.4

TARGET_TICKERS = [
    # Major Large Cap Biotech/Pharma
    "LLY", "NVO", "JNJ", "MRK", "ABBV", "PFE", "AMGN", "AZN", "NVS", "BMY",
    "GILD", "SNY", "GSK", "VRTX", "REGN", "BIIB", "ALNY", "SGEN", "INCY", "EXEL",
    
    # mRNA/Therapeutic Platforms
    "MRNA", "BNTX", "NVAX", "VKTX", "ARCT", "TBIO", "MESO", "PACB", "TWST", "DNA",
    
    # Gene Editing
    "CRSP", "EDIT", "NTLA", "BEAM", "VRTX", "BLUE", "FATE", "SGMO", "MYGN", "BOLD",
    
    # Oncology
    "RHHBY", "GMAB", "ARGX", "BGNE", "ZLAB", "HCM", "LPTX", "ARVN", "KROS", "MREO",
    "XENE", "PTCT", "FDMT", "ATRA", "CRNX", "CABA", "ACRS", "SANA", "AUTL", "ALLO",
    "ZYME", "GLUE", "ARWR", "DYN", "KNSA", "MGNX", "XNCR", "TGTX", "RETA", "AXSM",
    
    # Rare Disease
    "BMRN", "IONS", "QURE", "SARE", "PTGX", "ANAB", "FOLD", "RARE", "HALO", "NGM",
    
    # Immunology/Autoimmune
    "RCUS", "IMVT", "PRAX", "ACAD", "SAGE", "ITCI", "INVA", "VNDA", "ARQT", "KALV",
    
    # Cardiovascular/Metabolic
    "MDGL", "VKTX", "EWTX", "CRBP", "TNGX", "CCXI", "CPRX", "ARDX", "AKBA", "IRWD",
    
    # Neurology/CNS
    "BIIB", "NBIX", "SNPX", "ATXI", "AVXL", "ANVS", "SAVA", "RCKT", "CYTK", "ADVM",
    
    # Infectious Disease
    "DVAX", "ADMA", "JYNX", "ENOB", "TARA", "SCPH", "ACHL", "ATOS", "POAI",
    
    # Emerging/Small Cap Biotech
    "BEAM", "KYMR", "NTLA", "CRSP", "EDIT", "SRPT", "BPMC", "KRTX", "RVNC", "ZNTL",
    "RXRX", "ARHS", "CMPX", "DAWN", "ERAS", "EWTX", "HARP", "IMCR", "IPSC", "JANX",
    "KROS", "LVTX", "MAZE", "NRIX", "NUVB", "OMGA", "PRME", "QTRX", "RAIN", "RLAY",
    "RUBY", "SANA", "SCYX", "SEER", "SWTX", "TCRR", "TIL", "TNYA", "TRML", "UTRS",
    "VCEL", "VERA", "VIR", "VOR", "WVE", "XFOR", "YMAB", "ZEAL", "ZLAB", "ZNTL",
    
    # Additional Pharma
    "TEVA", "APLS", "PLRX", "AMLX", "KDNY", "TALS", "PASG", "IOVA", "CRVS", "ALT",
    "IMGN", "MOR", "CDMO", "KURA", "DCPH", "GLYC", "PDSB", "RAPT", "FULC", "STOK",
    "EQRX", "COGT", "ELEV", "GRPH", "ABCL", "RLYB", "OLMA", "RNA", "CORT", "CCCC",
    "THRD", "BMEA", "ATRC", "BCRX", "INSM", "MRUS", "KPTI", "SRRA", "ATOM", "PETQ",
    "ORTX", "VYGR", "TSHA", "MYOV", "MLTX", "IMMX", "DICE", "IMGO", "TRDA", "LXRX",
    "MRKR", "CNTA", "NRBO", "ICVX", "LUMO", "CLDX", "BDTX", "ANEB", "BLU", "CALT",
    "DTIL", "GTHX", "KRON", "LXEO", "MOLN", "OBSV", "RCKT", "RLMD", "RYTM", "SRRK",
    "TCDA", "TCRX", "TERN", "TPST", "TYRA", "VINC", "VMD", "XOMA", "YTOR", "ZEV",
]

# Ensure no duplicates
TARGET_TICKERS = list(dict.fromkeys(TARGET_TICKERS))  # Remove duplicates while preserving order

MANUAL_TICKER_OVERRIDES = {
    "Genentech": "RHHBY",
    "Roche": "RHHBY",
    "AstraZeneca": "AZN",
    "GlaxoSmithKline": "GSK",
    "Sanofi": "SNY",
    "Novartis": "NVS",
    "Pfizer": "PFE",
    "Johnson & Johnson": "JNJ",
    "Merck Sharp & Dohme": "MRK",
    "Eli Lilly": "LLY",
    "Bristol-Myers Squibb": "BMY",
    "AbbVie": "ABBV",
}
