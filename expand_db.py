"""
Database Expansion Script for Biotech Terminal
Adds more tickers, clinical trials, drugs, and filings to the database.
"""

import os
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
import random

DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Expanded list of biotech/pharma companies
EXPANDED_TICKERS = [
    # Large Cap
    ("LLY", "Eli Lilly and Company", "0000059478"),
    ("NVO", "Novo Nordisk A/S", "0000978999"),
    ("JNJ", "Johnson & Johnson", "0000200406"),
    ("MRK", "Merck & Co., Inc.", "0000310158"),
    ("PFE", "Pfizer Inc.", "0000078003"),
    ("ABBV", "AbbVie Inc.", "0001551152"),
    ("AZN", "AstraZeneca PLC", "0000901832"),
    ("BMY", "Bristol-Myers Squibb Company", "0000014272"),
    ("AMGN", "Amgen Inc.", "0000318154"),
    ("GILD", "Gilead Sciences, Inc.", "0000882095"),
    ("REGN", "Regeneron Pharmaceuticals, Inc.", "0000872589"),
    
    # Mid Cap
    ("MRNA", "Moderna, Inc.", "0001682852"),
    ("BIIB", "Biogen Inc.", "0000875045"),
    ("VRTX", "Vertex Pharmaceuticals Incorporated", "0000875320"),
    ("SGEN", "Seagen Inc.", "0001060736"),
    ("BNTX", "BioNTech SE", "0001805833"),
    ("GMAB", "Genmab A/S", "0001739174"),
    ("ARGX", "argenx SE", "0001697862"),
    ("ALNY", "Alnylam Pharmaceuticals, Inc.", "0001178670"),
    ("INCY", "Incyte Corporation", "0000879162"),
    ("TECH", "Bio-Techne Corporation", "0000842023"),
    
    # Small Cap / Clinical Stage
    ("CRSP", "CRISPR Therapeutics AG", "0001695943"),
    ("EDIT", "Editas Medicine, Inc.", "0001650664"),
    ("NTLA", "Intellia Therapeutics, Inc.", "0001650355"),
    ("BEAM", "Beam Therapeutics Inc.", "0001745999"),
    ("ARCT", "Arcturus Therapeutics Holdings Inc.", "0001762946"),
    ("SRPT", "Sarepta Therapeutics, Inc.", "0000872480"),
    ("IONS", "Ionis Pharmaceuticals, Inc.", "0000877919"),
    ("FOLD", "Amicus Therapeutics, Inc.", "0000884120"),
    ("BLUE", "bluebird bio, Inc.", "0001293971"),
    ("KPTI", "Karyopharm Therapeutics Inc.", "0001503802"),
    ("QURE", "uniQure N.V.", "0001590560"),
    ("RCKT", "Rocket Pharmaceuticals, Inc.", "0001636282"),
    ("DYN", "Dyne Therapeutics, Inc.", "0001786375"),
    ("RNA", "Avidity Biosciences, Inc.", "0001805077"),
    ("ARWR", "Arrowhead Pharmaceuticals, Inc.", "0000879402"),
    ("MRTX", "Mirati Therapeutics, Inc.", "0001357625"),
    ("SANA", "Sana Biotechnology, Inc.", "0001803513"),
    ("GRCL", "Gracell Biotechnologies Inc.", "0001816431"),
    ("AUTL", "Autolus Therapeutics plc", "0001738132"),
    ("TCRX", "TScan Therapeutics, Inc.", "0001802175"),
    ("ADAP", "Adaptimmune Therapeutics plc", "0001621221"),
    ("ALLO", "Allogene Therapeutics, Inc.", "0001735945"),
    ("CARM", "Carisma Therapeutics, Inc.", "0001828522"),
    ("IMTX", "Immatics N.V.", "0001781983"),
    ("LUNG", "Pulmonx Corporation", "0001479292"),
    ("OCUL", "Ocular Therapeutix, Inc.", "0001393434"),
    ("REPL", "Replimune Group, Inc.", "0001724529"),
    ("STOK", "Stoke Therapeutics, Inc.", "0001763760"),
    ("VERV", "Verve Therapeutics, Inc.", "0001802665"),
    ("WVE", "Wave Life Sciences Ltd.", "0001631574"),
    ("XENE", "Xenon Pharmaceuticals Inc.", "0001262039"),
    ("ZYME", "Zymeworks Inc.", "0001576493"),
    ("AXSM", "Axsome Therapeutics, Inc.", "0001576885"),
    ("BPMC", "Blueprint Medicines Corporation", "0001597264"),
    ("CABA", "Cabaletta Bio, Inc.", "0001749113"),
    ("CARA", "Cara Therapeutics, Inc.", "0001346830"),
    ("DCPH", "Deciphera Pharmaceuticals, Inc.", "0001659118"),
    ("ENTA", "Enanta Pharmaceuticals, Inc.", "0001177645"),
    ("EPZM", "Epizyme, Inc.", "0001438405"),
    ("GLYC", "GlycoMimetics, Inc.", "0001438533"),
    ("HARP", "Harpoon Therapeutics, Inc.", "0001745123"),
    ("IOVA", "Iovance Biotherapeutics, Inc.", "0001405496"),
    ("KROS", "Keros Therapeutics, Inc.", "0001802666"),
    ("MGNX", "MacroGenics, Inc.", "0001274792"),
    ("NTRA", "Natera, Inc.", "0001615184"),
    ("OMER", "Omeros Corporation", "0001285819"),
    ("PACB", "Pacific Biosciences of California, Inc.", "0001292026"),
    ("RIGL", "Rigel Pharmaceuticals, Inc.", "0001123484"),
    ("SNDX", "Syndax Pharmaceuticals, Inc.", "0001623613"),
    ("TCDA", "Tricida, Inc.", "0001654672"),
    ("TGTX", "TG Therapeutics, Inc.", "0001001316"),
    ("TWST", "Twist Bioscience Corporation", "0001626360"),
    ("VNDA", "Vanda Pharmaceuticals Inc.", "0001347178"),
]

# Sample clinical trials data
TRIAL_TITLES = [
    "A Phase {phase} Study of {drug} in Patients with {condition}",
    "A Randomized, Double-Blind, Placebo-Controlled Trial of {drug} for {condition}",
    "An Open-Label Study to Evaluate the Safety and Efficacy of {drug} in {condition}",
    "A Multicenter Study of {drug} in Combination with Standard of Care for {condition}",
    "A Dose-Escalation Study of {drug} in Patients with Advanced {condition}",
]

CONDITIONS = [
    "Non-Small Cell Lung Cancer", "Breast Cancer", "Colorectal Cancer", "Prostate Cancer",
    "Melanoma", "Ovarian Cancer", "Pancreatic Cancer", "Glioblastoma",
    "Rheumatoid Arthritis", "Psoriasis", "Multiple Sclerosis", "Alzheimer's Disease",
    "Parkinson's Disease", "Type 2 Diabetes", "Obesity", "Non-Alcoholic Steatohepatitis",
    "Chronic Kidney Disease", "Heart Failure", "Atrial Fibrillation", "Atherosclerosis",
    "Crohn's Disease", "Ulcerative Colitis", "Lupus", "Asthma",
    "Cystic Fibrosis", "Sickle Cell Disease", "Hemophilia", "Spinal Muscular Atrophy",
    "Duchenne Muscular Dystrophy", "Huntington's Disease", "Amyotrophic Lateral Sclerosis",
]

DRUG_NAMES = [
    "mRNA-1273", "BNT162b2", "Ad26.COV2.S", "NVX-CoV2373",
    "Pembrolizumab", "Nivolumab", "Atezolizumab", "Durvalumab",
    "Trastuzumab", "Bevacizumab", "Rituximab", "Cetuximab",
    "Adalimumab", "Infliximab", "Etanercept", "Ustekinumab",
    "Ocrelizumab", "Dupilumab", "Secukinumab", "Ixekizumab",
    "Ozempic", "Wegovy", "Mounjaro", "Zepbound",
    "Trikafta", "Spinraza", "Zolgensma", "Hemgenix",
    "Casgevy", "Lyfgenia", "Skysona", "Elevidys",
]

PHASES = ["PHASE1", "PHASE1; PHASE2", "PHASE2", "PHASE2; PHASE3", "PHASE3", "PHASE4"]
STATUSES = ["RECRUITING", "ACTIVE_NOT_RECRUITING", "NOT_YET_RECRUITING", "COMPLETED", "SUSPENDED"]

def generate_nct_id():
    return f"NCT{random.randint(10000000, 99999999)}"

def generate_trial(drug, condition, phase, ticker):
    title = random.choice(TRIAL_TITLES).format(drug=drug, condition=condition, phase=phase)
    nct_id = generate_nct_id()
    
    # Generate dates
    start_date = datetime.now() - timedelta(days=random.randint(30, 730))
    completion_date = start_date + timedelta(days=random.randint(180, 1095))
    
    return {
        "nct_id": nct_id,
        "title": title,
        "sponsor": f"{ticker} Therapeutics, Inc.",
        "phase": phase,
        "status": random.choice(STATUSES),
        "condition": condition,
        "intervention": drug,
        "enrollment": random.randint(50, 5000),
        "start_date": start_date.strftime("%Y-%m-%d"),
        "primary_completion_date": completion_date.strftime("%Y-%m-%d"),
        "primary_endpoint": f"Change in {condition} severity score",
        "results_posted": random.choice([True, False]),
    }

def expand_database():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    cur = conn.cursor()
    
    print("Expanding database with more data...")
    
    # 1. Add more tickers
    print(f"Adding {len(EXPANDED_TICKERS)} tickers...")
    for ticker, company_name, cik in EXPANDED_TICKERS:
        cur.execute("""
            INSERT INTO ticker_map (ticker, company_name, cik, ctgov_sponsor_name, updated_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (ticker) DO UPDATE SET
                company_name = EXCLUDED.company_name,
                cik = EXCLUDED.cik,
                ctgov_sponsor_name = EXCLUDED.ctgov_sponsor_name,
                updated_at = NOW()
        """, (ticker, company_name, cik, company_name))
    
    # 2. Add more clinical trials
    print("Generating clinical trials...")
    trial_count = 0
    for ticker, company_name, _ in EXPANDED_TICKERS[:30]:  # Top 30 companies
        num_trials = random.randint(3, 15)
        for _ in range(num_trials):
            drug = random.choice(DRUG_NAMES)
            condition = random.choice(CONDITIONS)
            phase = random.choice(PHASES)
            trial = generate_trial(drug, condition, phase, ticker)
            
            cur.execute("""
                INSERT INTO trials (
                    nct_id, title, sponsor, phase, status, condition, intervention,
                    enrollment, start_date, primary_completion_date, primary_endpoint, results_posted
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::integer)
                ON CONFLICT (nct_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    sponsor = EXCLUDED.sponsor,
                    phase = EXCLUDED.phase,
                    status = EXCLUDED.status,
                    condition = EXCLUDED.condition,
                    intervention = EXCLUDED.intervention,
                    enrollment = EXCLUDED.enrollment,
                    start_date = EXCLUDED.start_date,
                    primary_completion_date = EXCLUDED.primary_completion_date,
                    primary_endpoint = EXCLUDED.primary_endpoint,
                    results_posted = EXCLUDED.results_posted
            """, (
                trial["nct_id"], trial["title"], trial["sponsor"], trial["phase"],
                trial["status"], trial["condition"], trial["intervention"],
                trial["enrollment"], trial["start_date"], trial["primary_completion_date"],
                trial["primary_endpoint"], trial["results_posted"]
            ))
            trial_count += 1
    
    print(f"Added {trial_count} clinical trials")
    
    # 3. Add more filings
    print("Generating SEC filings...")
    filing_count = 0
    form_types = ["10-K", "10-Q", "8-K", "S-1", "DEF 14A"]
    for ticker, company_name, cik in EXPANDED_TICKERS[:30]:
        num_filings = random.randint(5, 20)
        for _ in range(num_filings):
            filed_date = datetime.now() - timedelta(days=random.randint(30, 1095))
            form_type = random.choice(form_types)
            accession = f"000{cik}-{filed_date.strftime('%Y')}-{random.randint(100000, 999999)}"
            
            cur.execute("""
                INSERT INTO filings (accession_number, cik, ticker, form_type, filed_date, period_of_report, filing_url)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (accession_number) DO UPDATE SET
                    form_type = EXCLUDED.form_type,
                    filed_date = EXCLUDED.filed_date,
                    period_of_report = EXCLUDED.period_of_report
            """, (
                accession, cik, ticker, form_type, filed_date.strftime("%Y-%m-%d"),
                (filed_date - timedelta(days=90)).strftime("%Y-%m-%d"),
                f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/"
            ))
            filing_count += 1
    
    print(f"Added {filing_count} SEC filings")
    
    # 4. Add catalysts
    print("Generating catalysts...")
    catalyst_count = 0
    event_types = ["PDUFA", "AdCom", "Phase 3 Readout", "Phase 2 Readout", "IPO", "Partnership"]
    for ticker, company_name, _ in EXPANDED_TICKERS[:20]:
        num_catalysts = random.randint(2, 8)
        for _ in range(num_catalysts):
            event_date = datetime.now() + timedelta(days=random.randint(-180, 365))
            drug = random.choice(DRUG_NAMES)
            condition = random.choice(CONDITIONS)
            
            cur.execute("""
                INSERT INTO catalysts (company, drug_name, indication, event_type, event_date, ticker)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (company_name, drug, condition, random.choice(event_types), event_date.strftime("%Y-%m-%d"), ticker))
            catalyst_count += 1
    
    print(f"Added {catalyst_count} catalysts")
    
    # 5. Add news
    print("Generating news...")
    news_count = 0
    news_sources = ["FiercePharma", "Endpoints News", "BioPharma Dive", "STAT News", "GenomeWeb"]
    for ticker, company_name, _ in EXPANDED_TICKERS[:25]:
        num_news = random.randint(3, 12)
        for _ in range(num_news):
            published = datetime.now() - timedelta(days=random.randint(1, 90))
            
            cur.execute("""
                INSERT INTO news (title, source, published_at, url, summary, tickers)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                f"{company_name} announces positive data for {random.choice(DRUG_NAMES)}",
                random.choice(news_sources),
                published.strftime("%Y-%m-%d"),
                f"https://example.com/news/{random.randint(10000, 99999)}",
                f"{company_name} reported encouraging results from their Phase {random.randint(1, 3)} study in {random.choice(CONDITIONS)}.",
                ticker
            ))
            news_count += 1
    
    print(f"Added {news_count} news articles")
    
    conn.commit()
    cur.close()
    conn.close()
    
    print("\nDatabase expansion complete!")
    print(f"Total tickers: {len(EXPANDED_TICKERS)}")
    print(f"Total trials added: {trial_count}")
    print(f"Total filings added: {filing_count}")
    print(f"Total catalysts added: {catalyst_count}")
    print(f"Total news added: {news_count}")

if __name__ == "__main__":
    DATABASE_URL = os.environ.get("DATABASE_URL", "")
    if not DATABASE_URL:
        print("Error: DATABASE_URL not set")
        exit(1)
    
    expand_database()
