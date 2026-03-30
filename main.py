import argparse
import logging
import sys
from datetime import datetime, timedelta
from config import TARGET_TICKERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("biotech-terminal")


def cmd_init():
    from models.ticker_map import init_db, build_ticker_map
    logger.info("Initializing database on Neon...")
    init_db()
    logger.info("Building ticker map from SEC EDGAR...")
    n = build_ticker_map()
    print(f"Database initialized on Neon. {n} tickers resolved.")


def cmd_ingest(source: str, tickers: list[str]):
    if source in ("all", "trials"):
        from ingestion.clinicaltrials import ingest_for_tickers
        results = ingest_for_tickers(tickers)
        print(f"ClinicalTrials: {sum(results.values())} trials upserted across {len(results)} tickers.")

    if source in ("all", "edgar"):
        from ingestion.edgar import ingest_for_tickers
        results = ingest_for_tickers(tickers)
        print(f"EDGAR: {sum(results.values())} filings upserted across {len(results)} tickers.")

    if source in ("all", "fda"):
        from ingestion.fda_calendar import ingest_all
        n = ingest_all()
        print(f"FDA Calendar: {n} events upserted.")

    if source in ("all", "pubmed"):
        from ingestion.pubmed import ingest_from_trials_table, ingest_for_tickers
        n1 = ingest_from_trials_table()
        n2 = sum(ingest_for_tickers(tickers).values())
        print(f"PubMed: {n1 + n2} publications upserted.")


def cmd_query_upcoming(days: int = 30):
    from models.ticker_map import get_connection
    conn = get_connection()
    cur = conn.cursor()
    cutoff = (datetime.utcnow() + timedelta(days=days)).strftime("%Y-%m-%d")
    today = datetime.utcnow().strftime("%Y-%m-%d")

    cur.execute(
        """SELECT event_date, event_type, company, drug_name, indication, ticker
           FROM catalysts
           WHERE event_date >= %s AND event_date <= %s
           ORDER BY event_date ASC""",
        (today, cutoff),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        print(f"No catalyst events in the next {days} days.")
        return

    print(f"\n{'DATE':<12} {'TYPE':<8} {'TICKER':<8} {'COMPANY':<30} {'DRUG':<25} {'INDICATION'}")
    print("-" * 110)
    for r in rows:
        print(
            f"{r['event_date']:<12} {r['event_type']:<8} {(r['ticker'] or '?'):<8} "
            f"{(r['company'] or '')[:29]:<30} {(r['drug_name'] or '')[:24]:<25} "
            f"{(r['indication'] or '')[:40]}"
        )


def cmd_query_pipeline(ticker: str):
    from models.ticker_map import get_connection, get_ticker_info
    info = get_ticker_info(ticker)
    if not info:
        print(f"Ticker {ticker} not found in database.")
        return

    company = info.get("company_name") or info.get("ctgov_sponsor_name") or ticker
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """SELECT nct_id, title, phase, status, condition, intervention,
                  primary_completion_date, results_posted
           FROM trials WHERE sponsor ILIKE %s
           ORDER BY phase DESC, status""",
        (f"%{company.split()[0]}%",),
    )
    trials = cur.fetchall()

    cur.execute(
        "SELECT form_type, filed_date, filing_url FROM filings WHERE ticker = %s ORDER BY filed_date DESC LIMIT 5",
        (ticker,),
    )
    filings = cur.fetchall()

    cur.execute(
        "SELECT event_date, event_type, drug_name, indication FROM catalysts WHERE ticker = %s ORDER BY event_date",
        (ticker,),
    )
    catalysts = cur.fetchall()

    cur.close()
    conn.close()

    print(f"\n=== {ticker} — {company} ===\n")

    print(f"PIPELINE ({len(trials)} trials):")
    print(f"{'NCT ID':<15} {'PHASE':<10} {'STATUS':<20} {'CONDITION':<30} {'COMPLETION'}")
    print("-" * 100)
    for t in trials:
        print(
            f"{t['nct_id']:<15} {(t['phase'] or '?'):<10} {(t['status'] or ''):<20} "
            f"{(t['condition'] or '')[:29]:<30} {t['primary_completion_date'] or ''}"
        )

    if catalysts:
        print(f"\nCATALYSTS ({len(catalysts)}):")
        for c in catalysts:
            print(f"  {c['event_date']}  {c['event_type']}  {c['drug_name'] or ''}  {c['indication'] or ''}")

    if filings:
        print(f"\nRECENT FILINGS ({len(filings)}):")
        for f in filings:
            print(f"  {f['filed_date']}  {f['form_type']}  {f['filing_url']}")


def cmd_status():
    from models.ticker_map import get_connection
    conn = get_connection()
    cur = conn.cursor()

    tables = ["trials", "catalysts", "filings", "publications", "ticker_map", "ingestion_log"]
    print("\n=== Database Status (Neon) ===")
    for t in tables:
        try:
            cur.execute(f"SELECT COUNT(*) AS cnt FROM {t}")
            count = cur.fetchone()["cnt"]
            print(f"  {t:<20} {count:>8} rows")
        except Exception:
            print(f"  {t:<20}  (table not found)")

    cur.execute(
        """SELECT source, ticker, rows_upserted, status, finished_at
           FROM ingestion_log ORDER BY finished_at DESC LIMIT 10"""
    )
    last_logs = cur.fetchall()
    cur.close()
    conn.close()

    if last_logs:
        print("\nRecent ingestion log:")
        for row in last_logs:
            print(f"  {str(row['finished_at']):<24}  {row['source']:<20} {(row['ticker'] or 'all'):<10} {row['rows_upserted']:>6} rows  [{row['status']}]")


def cmd_scheduler(tickers: list[str]):
    from ingestion.scheduler import start_background_scheduler
    start_background_scheduler(tickers)


def main():
    parser = argparse.ArgumentParser(
        description="Biotech Terminal — Data Ingestion CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Initialize Neon database and build ticker map")

    p_ingest = subparsers.add_parser("ingest", help="Ingest data from sources")
    p_ingest.add_argument(
        "source",
        choices=["all", "trials", "edgar", "fda", "pubmed"],
        help="Data source to ingest",
    )
    p_ingest.add_argument("--tickers", nargs="+", help="Specific tickers to ingest (default: all configured)")

    p_upcoming = subparsers.add_parser("upcoming", help="Show upcoming catalyst events")
    p_upcoming.add_argument("--days", type=int, default=30, help="Lookahead window in days (default: 30)")

    p_pipeline = subparsers.add_parser("pipeline", help="Show pipeline for a ticker")
    p_pipeline.add_argument("ticker", help="Ticker symbol (e.g., MRNA)")

    subparsers.add_parser("status", help="Show database row counts and recent ingestion log")

    p_scheduler = subparsers.add_parser("scheduler", help="Start background scheduled ingestion")
    p_scheduler.add_argument("--tickers", nargs="+", help="Specific tickers (default: all configured)")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init()
    elif args.command == "ingest":
        tickers = args.tickers or TARGET_TICKERS
        cmd_ingest(args.source, tickers)
    elif args.command == "upcoming":
        cmd_query_upcoming(args.days)
    elif args.command == "pipeline":
        cmd_query_pipeline(args.ticker.upper())
    elif args.command == "status":
        cmd_status()
    elif args.command == "scheduler":
        tickers = args.tickers or TARGET_TICKERS
        cmd_scheduler(tickers)


if __name__ == "__main__":
    main()
