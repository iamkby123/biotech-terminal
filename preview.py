import os
import sys
from dotenv import load_dotenv
load_dotenv()

import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta

DATABASE_URL = os.environ.get("DATABASE_URL", "")

def conn():
    return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"

def header(text):
    width = 110
    print(f"\n{BOLD}{CYAN}{'═' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'═' * width}{RESET}")

def section(text):
    print(f"\n{BOLD}{YELLOW}  ▸ {text}{RESET}")
    print(f"  {'─' * 106}")

def show_summary():
    header("BIOTECH TERMINAL — DATABASE SNAPSHOT")
    c = conn()
    cur = c.cursor()

    tables = ["trials", "catalysts", "filings", "publications", "ticker_map"]
    print(f"\n  {'TABLE':<20} {'ROWS':>8}   {'LAST INGESTED'}")
    print(f"  {'─'*60}")
    for t in tables:
        cur.execute(f"SELECT COUNT(*) AS cnt FROM {t}")
        count = cur.fetchone()["cnt"]
        try:
            cur.execute(f"SELECT MAX(ingested_at) AS last FROM {t}")
            last = cur.fetchone()["last"]
            last_str = str(last)[:16] if last else "—"
        except Exception:
            last_str = "—"
        bar_len = min(30, count // 10)
        bar = f"{GREEN}{'█' * bar_len}{DIM}{'░' * (30 - bar_len)}{RESET}"
        print(f"  {t:<20} {count:>8}   {last_str:<20} {bar}")

    cur.close()
    c.close()

def show_trials_breakdown():
    section("TRIALS BY PHASE")
    c = conn()
    cur = c.cursor()
    cur.execute("""
        SELECT phase, status, COUNT(*) AS cnt
        FROM trials
        GROUP BY phase, status
        ORDER BY phase DESC NULLS LAST, cnt DESC
        LIMIT 20
    """)
    rows = cur.fetchall()
    print(f"  {'PHASE':<25} {'STATUS':<25} {'COUNT':>6}")
    print(f"  {'─'*60}")
    for r in rows:
        phase = (r["phase"] or "Unknown")[:24]
        status = (r["status"] or "Unknown")[:24]
        count = r["cnt"]
        color = GREEN if "RECRUITING" in status.upper() else (YELLOW if "ACTIVE" in status.upper() else DIM)
        print(f"  {phase:<25} {color}{status:<25}{RESET} {count:>6}")
    cur.close()
    c.close()

def show_top_sponsors():
    section("TOP SPONSORS BY TRIAL COUNT")
    c = conn()
    cur = c.cursor()
    cur.execute("""
        SELECT sponsor, COUNT(*) AS cnt,
               SUM(CASE WHEN status = 'RECRUITING' THEN 1 ELSE 0 END) AS recruiting,
               SUM(CASE WHEN results_posted = 1 THEN 1 ELSE 0 END) AS with_results
        FROM trials
        WHERE sponsor IS NOT NULL
        GROUP BY sponsor
        ORDER BY cnt DESC
        LIMIT 15
    """)
    rows = cur.fetchall()
    print(f"  {'SPONSOR':<40} {'TOTAL':>6}  {'RECRUITING':>10}  {'W/ RESULTS':>10}")
    print(f"  {'─'*75}")
    for r in rows:
        print(f"  {(r['sponsor'] or '')[:39]:<40} {r['cnt']:>6}  {r['recruiting']:>10}  {r['with_results']:>10}")
    cur.close()
    c.close()

def show_recent_filings():
    section("RECENT SEC FILINGS (last 10)")
    c = conn()
    cur = c.cursor()
    cur.execute("""
        SELECT ticker, form_type, filed_date, filing_url
        FROM filings
        ORDER BY filed_date DESC NULLS LAST
        LIMIT 10
    """)
    rows = cur.fetchall()
    print(f"  {'TICKER':<8} {'FORM':<8} {'DATE':<12} {'URL'}")
    print(f"  {'─'*100}")
    for r in rows:
        url = (r["filing_url"] or "")[:60]
        print(f"  {(r['ticker'] or ''):<8} {(r['form_type'] or ''):<8} {(r['filed_date'] or ''):<12} {url}")
    cur.close()
    c.close()

def show_upcoming_catalysts():
    section("UPCOMING CATALYSTS (next 90 days)")
    c = conn()
    cur = c.cursor()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    cutoff = (datetime.utcnow() + timedelta(days=90)).strftime("%Y-%m-%d")
    cur.execute("""
        SELECT event_date, event_type, company, drug_name, indication, ticker
        FROM catalysts
        WHERE event_date >= %s AND event_date <= %s
        ORDER BY event_date ASC
        LIMIT 20
    """, (today, cutoff))
    rows = cur.fetchall()
    if not rows:
        print(f"  {DIM}No upcoming events found (run: python main.py ingest fda){RESET}")
    else:
        print(f"  {'DATE':<12} {'TYPE':<8} {'TICKER':<8} {'COMPANY':<28} {'DRUG':<22} {'INDICATION'}")
        print(f"  {'─'*100}")
        for r in rows:
            print(
                f"  {(r['event_date'] or ''):<12} {(r['event_type'] or ''):<8} "
                f"{(r['ticker'] or '?'):<8} {(r['company'] or '')[:27]:<28} "
                f"{(r['drug_name'] or '')[:21]:<22} {(r['indication'] or '')[:35]}"
            )
    cur.close()
    c.close()

def show_pipeline(ticker: str):
    header(f"PIPELINE VIEW — {ticker.upper()}")
    c = conn()
    cur = c.cursor()

    cur.execute("SELECT * FROM ticker_map WHERE ticker = %s", (ticker.upper(),))
    info = cur.fetchone()
    if not info:
        print(f"  {RED}Ticker {ticker} not found in database.{RESET}")
        c.close()
        return

    company = info.get("company_name") or ticker
    print(f"\n  Company : {BOLD}{company}{RESET}")
    print(f"  CIK     : {info.get('cik') or '—'}")
    print(f"  CTGOV   : {info.get('ctgov_sponsor_name') or '—'}")

    cur.execute("""
        SELECT nct_id, title, phase, status, condition, intervention,
               primary_completion_date, results_posted
        FROM trials WHERE sponsor ILIKE %s
        ORDER BY phase DESC NULLS LAST, status
    """, (f"%{company.split()[0]}%",))
    trials = cur.fetchall()

    section(f"CLINICAL TRIALS ({len(trials)})")
    if trials:
        print(f"  {'NCT ID':<14} {'PHASE':<18} {'STATUS':<22} {'CONDITION':<30} {'COMPLETION'}")
        print(f"  {'─'*100}")
        for t in trials:
            status = t["status"] or ""
            color = GREEN if "RECRUITING" in status.upper() else (YELLOW if "ACTIVE" in status.upper() else DIM)
            results_flag = f" {GREEN}[R]{RESET}" if t["results_posted"] else ""
            print(
                f"  {t['nct_id']:<14} {(t['phase'] or '?')[:17]:<18} "
                f"{color}{status[:21]:<22}{RESET} "
                f"{(t['condition'] or '')[:29]:<30} "
                f"{t['primary_completion_date'] or '—'}{results_flag}"
            )

    cur.execute("""
        SELECT form_type, filed_date, filing_url FROM filings
        WHERE ticker = %s ORDER BY filed_date DESC LIMIT 5
    """, (ticker.upper(),))
    filings = cur.fetchall()
    section(f"SEC FILINGS ({len(filings)})")
    for f in filings:
        print(f"  {(f['filed_date'] or ''):<12}  {(f['form_type'] or ''):<8}  {(f['filing_url'] or '')[:70]}")

    cur.execute("""
        SELECT event_date, event_type, drug_name, indication FROM catalysts
        WHERE ticker = %s ORDER BY event_date
    """, (ticker.upper(),))
    cats = cur.fetchall()
    section(f"CATALYSTS ({len(cats)})")
    for c_row in cats:
        print(f"  {(c_row['event_date'] or ''):<12}  {(c_row['event_type'] or ''):<8}  {(c_row['drug_name'] or ''):<25}  {(c_row['indication'] or '')[:40]}")

    cur.execute("""
        SELECT p.pmid, p.title, p.pub_date, p.journal
        FROM publications p
        JOIN trials t ON p.nct_id = t.nct_id
        WHERE t.sponsor ILIKE %s
        ORDER BY p.pub_date DESC NULLS LAST
        LIMIT 5
    """, (f"%{company.split()[0]}%",))
    pubs = cur.fetchall()
    section(f"RECENT PUBLICATIONS ({len(pubs)})")
    for p in pubs:
        print(f"  {(p['pub_date'] or ''):<12}  {(p['journal'] or '')[:25]:<27}  {(p['title'] or '')[:65]}")

    cur.close()
    c.close()

def show_ingestion_log():
    section("INGESTION LOG (last 15)")
    c = conn()
    cur = c.cursor()
    cur.execute("""
        SELECT source, ticker, rows_upserted, status, finished_at
        FROM ingestion_log ORDER BY finished_at DESC LIMIT 15
    """)
    rows = cur.fetchall()
    print(f"  {'FINISHED':<22} {'SOURCE':<20} {'TICKER':<10} {'ROWS':>6}  STATUS")
    print(f"  {'─'*70}")
    for r in rows:
        color = GREEN if r["status"] == "ok" else RED
        print(f"  {str(r['finished_at'])[:19]:<22} {(r['source'] or ''):<20} {(r['ticker'] or 'all'):<10} {r['rows_upserted']:>6}  {color}{r['status']}{RESET}")
    cur.close()
    c.close()

def main():
    args = sys.argv[1:]

    if not args or args[0] == "summary":
        show_summary()
        show_trials_breakdown()
        show_top_sponsors()
        show_recent_filings()
        show_upcoming_catalysts()
        show_ingestion_log()
        print()

    elif args[0] == "pipeline" and len(args) > 1:
        show_pipeline(args[1])
        print()

    elif args[0] == "log":
        show_ingestion_log()
        print()

    else:
        print(f"Usage:")
        print(f"  python preview.py              — full dashboard")
        print(f"  python preview.py pipeline MRNA — single company")
        print(f"  python preview.py log           — ingestion log")

if __name__ == "__main__":
    main()
