"""
Options chain + insider signal service.
Sources: yfinance (options chain, insider transactions) + Finnhub (insider sentiment).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import math
import yfinance as yf
import httpx

from cache import TTLCache
from source_status import SourceStatus

logger = logging.getLogger(__name__)

options_cache = TTLCache(max_size=100, default_ttl=300)   # 5 min
insider_cache = TTLCache(max_size=100, default_ttl=600)   # 10 min

def _safe_float(v, default=0.0):
    """Convert value to float, replacing NaN/inf with default."""
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default

def _safe_int(v, default=0):
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return default
        return int(f)
    except (TypeError, ValueError):
        return default

FINNHUB_API_KEY = "ct2qh1pr01qhb1amkn3gct2qh1pr01qhb1amkn40"
FINNHUB_BASE = "https://finnhub.io/api/v1"


def fetch_options_chain(ticker: str) -> tuple[Optional[dict], SourceStatus]:
    cached = options_cache.get(ticker)
    if cached is not None:
        value, meta = cached
        return value, SourceStatus.ok("yfinance_options", 1, data_mode="cached",
                                       cache_hit=True, ttl_remaining=meta.ttl_remaining)

    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return {"ticker": ticker, "has_data": False, "signal": "no_data",
                    "message": "No options data available"}, SourceStatus.empty("yfinance_options")

        # Use nearest expiration for analysis
        nearest_exp = expirations[0]
        chain = stock.option_chain(nearest_exp)
        calls = chain.calls
        puts = chain.puts

        if calls.empty and puts.empty:
            return {"ticker": ticker, "has_data": False, "signal": "no_data",
                    "message": "Options chain empty"}, SourceStatus.empty("yfinance_options")

        # Compute metrics
        total_call_vol = int(calls["volume"].sum()) if "volume" in calls.columns else 0
        total_put_vol = int(puts["volume"].sum()) if "volume" in puts.columns else 0
        total_call_oi = int(calls["openInterest"].sum()) if "openInterest" in calls.columns else 0
        total_put_oi = int(puts["openInterest"].sum()) if "openInterest" in puts.columns else 0

        put_call_ratio = round(total_put_vol / max(total_call_vol, 1), 2)
        put_call_oi_ratio = round(total_put_oi / max(total_call_oi, 1), 2)

        # Average implied volatility
        avg_call_iv = round(float(calls["impliedVolatility"].mean()), 4) if "impliedVolatility" in calls.columns and not calls["impliedVolatility"].isna().all() else None
        avg_put_iv = round(float(puts["impliedVolatility"].mean()), 4) if "impliedVolatility" in puts.columns and not puts["impliedVolatility"].isna().all() else None
        avg_iv = None
        if avg_call_iv is not None and avg_put_iv is not None:
            avg_iv = round((avg_call_iv + avg_put_iv) / 2, 4)
        elif avg_call_iv is not None:
            avg_iv = avg_call_iv
        elif avg_put_iv is not None:
            avg_iv = avg_put_iv

        # Unusual activity detection
        unusual_puts = []
        if "volume" in puts.columns and "openInterest" in puts.columns:
            for _, row in puts.iterrows():
                vol = row.get("volume", 0) or 0
                oi = row.get("openInterest", 0) or 0
                if oi > 0 and vol > 2 * oi:
                    unusual_puts.append({
                        "strike": float(row.get("strike", 0)),
                        "volume": int(vol),
                        "open_interest": int(oi),
                        "vol_oi_ratio": round(vol / oi, 1),
                    })

        unusual_calls = []
        if "volume" in calls.columns and "openInterest" in calls.columns:
            for _, row in calls.iterrows():
                vol = row.get("volume", 0) or 0
                oi = row.get("openInterest", 0) or 0
                if oi > 0 and vol > 2 * oi:
                    unusual_calls.append({
                        "strike": float(row.get("strike", 0)),
                        "volume": int(vol),
                        "open_interest": int(oi),
                        "vol_oi_ratio": round(vol / oi, 1),
                    })

        # Overall signal
        if put_call_ratio > 1.5:
            signal = "bearish"
        elif put_call_ratio < 0.5:
            signal = "bullish"
        else:
            signal = "neutral"

        # Boost signal if unusual activity detected
        if len(unusual_puts) >= 3:
            signal = "bearish"
        elif len(unusual_calls) >= 3:
            signal = "bullish"

        result = {
            "ticker": ticker,
            "has_data": True,
            "expiration": nearest_exp,
            "expirations_available": len(expirations),
            "put_call_volume_ratio": put_call_ratio,
            "put_call_oi_ratio": put_call_oi_ratio,
            "total_call_volume": total_call_vol,
            "total_put_volume": total_put_vol,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "avg_implied_volatility": avg_iv,
            "avg_call_iv": avg_call_iv,
            "avg_put_iv": avg_put_iv,
            "unusual_put_activity": unusual_puts[:5],
            "unusual_call_activity": unusual_calls[:5],
            "signal": signal,
        }

        options_cache.set(ticker, result)
        return result, SourceStatus.ok("yfinance_options", 1, data_mode="live")

    except Exception as e:
        logger.warning(f"Options fetch failed for {ticker}: {e}")
        return {"ticker": ticker, "has_data": False, "signal": "no_data",
                "error": str(e)}, SourceStatus.error("yfinance_options", str(e))


def fetch_insider_activity(ticker: str) -> tuple[Optional[dict], SourceStatus]:
    cached = insider_cache.get(ticker)
    if cached is not None:
        value, meta = cached
        return value, SourceStatus.ok("insider", 1, data_mode="cached",
                                       cache_hit=True, ttl_remaining=meta.ttl_remaining)

    transactions = []
    yf_status = None
    finnhub_status = None

    # --- yfinance insider transactions ---
    try:
        stock = yf.Ticker(ticker)
        insider_df = stock.insider_transactions
        if insider_df is not None and not insider_df.empty:
            for _, row in insider_df.head(20).iterrows():
                transactions.append({
                    "source": "yfinance",
                    "insider": str(row.get("Insider", row.get("insider", "")) or ""),
                    "relation": str(row.get("Relation", row.get("relation", "")) or ""),
                    "transaction": str(row.get("Transaction", row.get("transaction", "")) or ""),
                    "shares": _safe_int(row.get("Shares", row.get("shares", 0))),
                    "value": _safe_float(row.get("Value", row.get("value", 0))),
                    "date": str(row.get("Start Date", row.get("startDate", row.get("date", ""))) or ""),
                })
            yf_status = SourceStatus.ok("yfinance_insider", len(transactions))
        else:
            yf_status = SourceStatus.empty("yfinance_insider")
    except Exception as e:
        logger.warning(f"yfinance insider failed for {ticker}: {e}")
        yf_status = SourceStatus.error("yfinance_insider", str(e))

    # --- Finnhub insider sentiment ---
    finnhub_sentiment = None
    try:
        if FINNHUB_API_KEY and not FINNHUB_API_KEY.startswith("ct2qh1"):
            r = httpx.get(
                f"{FINNHUB_BASE}/stock/insider-sentiment",
                params={"symbol": ticker, "token": FINNHUB_API_KEY, "from": "2025-01-01"},
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                if data.get("data"):
                    recent = data["data"][-3:]  # last 3 months
                    total_change = sum(m.get("change", 0) for m in recent)
                    total_mspr = sum(m.get("mspr", 0) for m in recent)
                    finnhub_sentiment = {
                        "months_analyzed": len(recent),
                        "net_share_change": total_change,
                        "mspr": round(total_mspr, 4),
                        "signal": "selling" if total_mspr < -5 else "buying" if total_mspr > 5 else "neutral",
                    }
                    finnhub_status = SourceStatus.ok("finnhub_insider", len(recent))
                else:
                    finnhub_status = SourceStatus.empty("finnhub_insider")
            else:
                finnhub_status = SourceStatus.error("finnhub_insider", f"HTTP {r.status_code}")
        else:
            finnhub_status = SourceStatus.empty("finnhub_insider", data_mode="skipped")
    except Exception as e:
        logger.warning(f"Finnhub insider failed for {ticker}: {e}")
        finnhub_status = SourceStatus.error("finnhub_insider", str(e))

    # Compute summary
    total_buys = sum(1 for t in transactions if "purchase" in t.get("transaction", "").lower() or "buy" in t.get("transaction", "").lower())
    total_sells = sum(1 for t in transactions if "sale" in t.get("transaction", "").lower() or "sell" in t.get("transaction", "").lower())
    buy_value = sum(_safe_float(t.get("value", 0)) for t in transactions if "purchase" in t.get("transaction", "").lower() or "buy" in t.get("transaction", "").lower())
    sell_value = sum(_safe_float(t.get("value", 0)) for t in transactions if "sale" in t.get("transaction", "").lower() or "sell" in t.get("transaction", "").lower())

    if sell_value > buy_value * 3:
        overall_signal = "bearish"
    elif buy_value > sell_value * 2:
        overall_signal = "bullish"
    else:
        overall_signal = "neutral"

    # Override with Finnhub sentiment if available
    if finnhub_sentiment and finnhub_sentiment["signal"] == "selling":
        overall_signal = "bearish"
    elif finnhub_sentiment and finnhub_sentiment["signal"] == "buying" and overall_signal != "bearish":
        overall_signal = "bullish"

    result = {
        "ticker": ticker,
        "has_data": len(transactions) > 0 or finnhub_sentiment is not None,
        "transactions": transactions[:10],
        "total_buys": total_buys,
        "total_sells": total_sells,
        "buy_value": round(buy_value, 2),
        "sell_value": round(sell_value, 2),
        "finnhub_sentiment": finnhub_sentiment,
        "signal": overall_signal,
    }

    insider_cache.set(ticker, result)
    statuses = [s for s in [yf_status, finnhub_status] if s is not None]
    return result, statuses[0] if len(statuses) == 1 else SourceStatus.ok("insider", 1, data_mode="live")
