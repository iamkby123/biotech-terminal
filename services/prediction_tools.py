"""
Prediction Tool-Use Framework.
Tools the prediction model can call to gather additional signals before making its prediction.
Each tool returns a dict with structured data + a one-line summary for the model.
"""
import logging
import json
from models.ticker_map import get_connection

logger = logging.getLogger(__name__)

PREDICTION_TOOLS: dict[str, dict] = {}


def _register(name, description, parameters):
    def decorator(fn):
        PREDICTION_TOOLS[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "execute": fn,
        }
        return fn
    return decorator


def get_tool_descriptions_for_prompt() -> str:
    lines = ["Available tools (call by outputting JSON: {\"tool_call\":\"name\",\"args\":{...}}):\n"]
    for name, tool in PREDICTION_TOOLS.items():
        lines.append(f"- {name}: {tool['description']}")
        if tool["parameters"]:
            lines.append(f"  Args: {tool['parameters']}")
    return "\n".join(lines)


def execute_tool(name: str, args: dict) -> dict:
    tool = PREDICTION_TOOLS.get(name)
    if not tool:
        return {"error": f"Unknown tool: {name}", "summary": f"Tool '{name}' not found."}
    try:
        return tool["execute"](**args)
    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        return {"error": str(e), "summary": f"Tool {name} failed: {e}"}


# ── Tool 1: Stock Data ──────────────────────────────────────────
@_register(
    "get_stock_data",
    "Get current stock price, change%, volume, market cap for a ticker",
    '{"ticker": "string"}',
)
def get_stock_data(ticker: str, **_) -> dict:
    from services.stock import fetch_stock_quote
    quote, status = fetch_stock_quote(ticker)
    if quote is None:
        return {"error": "No stock data", "summary": f"Could not fetch stock data for {ticker}."}

    d = quote.to_api_dict()
    price = d.get("price", "?")
    change = d.get("change_percent", "?")
    mcap = d.get("market_cap", "?")
    vol = d.get("volume", "?")

    # Format market cap
    if isinstance(mcap, (int, float)) and mcap > 0:
        if mcap >= 1e9:
            mcap_str = f"${mcap/1e9:.1f}B"
        elif mcap >= 1e6:
            mcap_str = f"${mcap/1e6:.0f}M"
        else:
            mcap_str = f"${mcap:,.0f}"
    else:
        mcap_str = "N/A"

    return {
        "data": d,
        "summary": f"Stock: ${price} ({change:+.1f}%), Market Cap: {mcap_str}, Volume: {vol:,}" if isinstance(change, (int, float)) else f"Stock: ${price}, Market Cap: {mcap_str}",
    }


# ── Tool 2: Options Signal ──────────────────────────────────────
@_register(
    "get_options_signal",
    "Get options chain analysis: put/call ratio, unusual activity, implied volatility, bearish/bullish signal",
    '{"ticker": "string"}',
)
def get_options_signal(ticker: str, **_) -> dict:
    from services.options import fetch_options_chain
    data, status = fetch_options_chain(ticker)
    if not data or not data.get("has_data"):
        return {"data": data, "summary": f"No options data available for {ticker}."}

    pcr = data.get("put_call_volume_ratio", "?")
    iv = data.get("avg_implied_volatility", "?")
    signal = data.get("signal", "?")
    unusual_puts = len(data.get("unusual_put_activity", []))
    unusual_calls = len(data.get("unusual_call_activity", []))

    summary = f"Options signal: {signal.upper()}. Put/Call ratio: {pcr}"
    if iv:
        summary += f", Avg IV: {iv*100:.1f}%"
    if unusual_puts:
        summary += f", {unusual_puts} unusual put strikes"
    if unusual_calls:
        summary += f", {unusual_calls} unusual call strikes"

    return {"data": data, "summary": summary}


# ── Tool 3: Insider Activity ────────────────────────────────────
@_register(
    "get_insider_activity",
    "Get insider trading transactions and sentiment: buying/selling patterns, unusual activity",
    '{"ticker": "string"}',
)
def get_insider_activity(ticker: str, **_) -> dict:
    from services.options import fetch_insider_activity
    data, status = fetch_insider_activity(ticker)
    if not data or not data.get("has_data"):
        return {"data": data, "summary": f"No insider activity data for {ticker}."}

    buys = data.get("total_buys", 0)
    sells = data.get("total_sells", 0)
    buy_val = data.get("buy_value", 0)
    sell_val = data.get("sell_value", 0)
    signal = data.get("signal", "neutral")

    summary = f"Insider signal: {signal.upper()}. {buys} buys (${buy_val:,.0f}), {sells} sells (${sell_val:,.0f})"
    fh = data.get("finnhub_sentiment")
    if fh:
        summary += f". Finnhub MSPR: {fh.get('mspr', 0):.2f} ({fh.get('signal', '?')})"

    return {"data": data, "summary": summary}


# ── Tool 4: Recent News ─────────────────────────────────────────
@_register(
    "get_recent_news",
    "Get latest news articles with sentiment analysis for a ticker",
    '{"ticker": "string", "limit": "optional int, default 5"}',
)
def get_recent_news(ticker: str, limit: int = 5, **_) -> dict:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT title, source, published_at, sentiment, catalyst_type
            FROM news_articles
            WHERE ticker = %s
            ORDER BY published_at DESC LIMIT %s
        """, (ticker.upper(), min(int(limit), 10)))
        rows = [dict(r) for r in cur.fetchall()]
        if not rows:
            return {"data": [], "summary": f"No recent news found for {ticker}."}

        sentiments = [r.get("sentiment", "neutral") for r in rows]
        pos = sentiments.count("positive")
        neg = sentiments.count("negative")
        catalysts = [r.get("catalyst_type") for r in rows if r.get("catalyst_type")]

        summary = f"{len(rows)} recent articles. Sentiment: {pos} positive, {neg} negative."
        if catalysts:
            summary += f" Catalysts: {', '.join(set(catalysts)[:3])}"

        return {"data": rows, "summary": summary}
    except Exception as e:
        return {"data": [], "summary": f"News query failed: {e}"}
    finally:
        cur.close()
        conn.close()


# ── Tool 5: Sponsor History ─────────────────────────────────────
@_register(
    "get_sponsor_history",
    "Get sponsor's trial track record: completion rates, terminated trials, therapeutic areas",
    '{"sponsor_name": "string"}',
)
def get_sponsor_history(sponsor_name: str, **_) -> dict:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT status, COUNT(*) as cnt
            FROM trials
            WHERE sponsor ILIKE %s
            GROUP BY status ORDER BY cnt DESC
        """, (f"%{sponsor_name}%",))
        status_counts = {r["status"]: r["cnt"] for r in cur.fetchall()}
        total = sum(status_counts.values())
        if total == 0:
            return {"data": {}, "summary": f"No trial history found for sponsor '{sponsor_name}'."}

        completed = sum(v for k, v in status_counts.items() if "COMPLET" in k.upper())
        terminated = sum(v for k, v in status_counts.items() if "TERMINAT" in k.upper())
        withdrawn = sum(v for k, v in status_counts.items() if "WITHDRAW" in k.upper())

        comp_rate = completed / total * 100
        fail_rate = (terminated + withdrawn) / total * 100

        # Get phase distribution
        cur.execute("""
            SELECT phase, COUNT(*) as cnt FROM trials
            WHERE sponsor ILIKE %s GROUP BY phase ORDER BY phase
        """, (f"%{sponsor_name}%",))
        phases = {r["phase"]: r["cnt"] for r in cur.fetchall()}

        data = {
            "sponsor": sponsor_name,
            "total_trials": total,
            "completed": completed,
            "terminated": terminated,
            "withdrawn": withdrawn,
            "completion_rate": round(comp_rate, 1),
            "failure_rate": round(fail_rate, 1),
            "status_breakdown": status_counts,
            "phase_distribution": phases,
        }

        summary = (f"Sponsor '{sponsor_name}': {total} trials, {comp_rate:.0f}% completion rate, "
                   f"{fail_rate:.0f}% failure rate ({terminated} terminated, {withdrawn} withdrawn)")

        return {"data": data, "summary": summary}
    except Exception as e:
        return {"data": {}, "summary": f"Sponsor history query failed: {e}"}
    finally:
        cur.close()
        conn.close()


# ── Tool 6: Adverse Events ──────────────────────────────────────
@_register(
    "get_adverse_events",
    "Get adverse event summary for a trial: SAE count, deaths, discontinuation rate",
    '{"nct_id": "string"}',
)
def get_adverse_events(nct_id: str, **_) -> dict:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT sae_count, deaths_reported, discontinuation_rate, dose_limiting_toxicity
            FROM v2.adverse_event_summary WHERE nct_id = %s
        """, (nct_id,))
        row = cur.fetchone()
        if not row:
            return {"data": {}, "summary": f"No adverse event data available for {nct_id}."}

        data = dict(row)
        parts = []
        if data.get("sae_count"):
            parts.append(f"{data['sae_count']} SAEs")
        if data.get("deaths_reported"):
            parts.append(f"{data['deaths_reported']} deaths")
        if data.get("discontinuation_rate"):
            parts.append(f"{data['discontinuation_rate']*100:.1f}% discontinuation")
        if data.get("dose_limiting_toxicity"):
            parts.append("dose-limiting toxicity reported")

        summary = f"AE data for {nct_id}: {', '.join(parts)}" if parts else f"No significant AEs reported for {nct_id}."
        return {"data": data, "summary": summary}
    except Exception as e:
        return {"data": {}, "summary": f"AE query failed: {e}"}
    finally:
        cur.close()
        conn.close()


# ── Tool 7: Competitive Landscape ───────────────────────────────
@_register(
    "get_competitive_landscape",
    "Get competing trials and sponsors in the same indication/condition",
    '{"condition": "string", "exclude_nct_id": "optional string"}',
)
def get_competitive_landscape(condition: str, exclude_nct_id: str = None, **_) -> dict:
    conn = get_connection()
    cur = conn.cursor()
    try:
        query = """
            SELECT sponsor, phase, status, COUNT(*) as trial_count
            FROM trials
            WHERE condition ILIKE %s
        """
        params = [f"%{condition}%"]
        if exclude_nct_id:
            query += " AND nct_id != %s"
            params.append(exclude_nct_id)
        query += " GROUP BY sponsor, phase, status ORDER BY trial_count DESC LIMIT 20"

        cur.execute(query, params)
        rows = [dict(r) for r in cur.fetchall()]
        if not rows:
            return {"data": [], "summary": f"No competing trials found for condition '{condition}'."}

        sponsors = list(set(r["sponsor"] for r in rows))
        total = sum(r["trial_count"] for r in rows)

        summary = f"Competition in '{condition}': {total} trials from {len(sponsors)} sponsors. Top: {', '.join(sponsors[:3])}"
        return {"data": rows, "summary": summary}
    except Exception as e:
        return {"data": [], "summary": f"Competition query failed: {e}"}
    finally:
        cur.close()
        conn.close()


# ── Tool 8: Population Simulation ────────────────────────────────
@_register(
    "get_population_simulation",
    "Get drug demand simulation results: patient population characteristics, demand score, market penetration estimate",
    '{"drug_name": "string", "condition": "string", "disease_group": "string"}',
)
def get_population_simulation(drug_name: str, condition: str, disease_group: str = "other", **_) -> dict:
    from services.mirofish_sim import get_cached_simulation, run_demand_simulation
    result = get_cached_simulation(drug_name, condition, disease_group)
    if result is None:
        # Run a quick local simulation (100 agents for speed)
        result = run_demand_simulation(drug_name, condition, disease_group, population_size=100)

    if not result:
        return {"data": {}, "summary": f"No simulation data for {drug_name} in {condition}."}

    demand = result.get("demand", {})
    pop = result.get("population", {})
    score = demand.get("demand_score", "?")
    y1 = demand.get("year1_penetration_pct", "?")
    seeking = pop.get("treatment_seeking_pct", "?")

    summary = f"Demand simulation: score {score}/100, Y1 penetration {y1}%, {seeking}% actively seeking treatment"
    return {"data": result, "summary": summary}
