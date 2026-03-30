"""
Agent loop — orchestrates LLM calls and tool execution.
"""
import json
import logging
import re
import time

from agent import llm
from agent.tools import TOOL_REGISTRY, get_tool_descriptions

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5

SYSTEM_PROMPT = """You are a biotech clinical trial data-gathering agent. Your job is to search databases and APIs to find factual data about clinical trials, biotech companies, and related news.

IMPORTANT RULES:
- You NEVER predict or speculate about trial outcomes
- You NEVER make up data — always use tools
- Call one tool at a time
- After gathering enough data, produce your final answer

AVAILABLE TOOLS:

{tools}

TO CALL A TOOL, respond with ONLY this JSON (no other text):
{{"tool": "tool_name", "args": {{"param1": "value1"}}}}

WHEN YOU HAVE ENOUGH DATA, respond with ONLY this JSON (no other text):
{{"done": true, "summary": "Brief summary of findings", "results": [<data you gathered>]}}

Do not include any text outside the JSON object. Do not use markdown code blocks."""


SOURCE_MAP = {
    "search_trials_db": "db:trials",
    "get_trial_detail": "db:trials",
    "search_clinicaltrials_gov": "api:clinicaltrials.gov",
    "search_news": "db:news",
    "get_company_info": "db:ticker_map",
    "search_executive_bio": "web:duckduckgo",
}


def _try_parse_json(text: str) -> dict | None:
    """Try to extract a JSON object from text. Multiple strategies."""
    text = text.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: extract from markdown code block
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: find first { to last }
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        try:
            return json.loads(text[first : last + 1])
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def run_agent(query: str) -> dict:
    """Execute the agent loop. Returns structured response dict."""
    start = time.time()

    # Build system prompt with tool descriptions
    system_msg = SYSTEM_PROMPT.format(tools=get_tool_descriptions())
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": query},
    ]

    sources = set()
    tool_calls_made = 0
    all_tool_results = []
    final_parsed = None

    for iteration in range(MAX_ITERATIONS):
        logger.info("Agent iteration %d/%d", iteration + 1, MAX_ITERATIONS)

        try:
            response = llm.chat(messages)
        except RuntimeError as e:
            logger.error("LLM error: %s", e)
            return _error_response(query, str(e), sources, tool_calls_made, start)

        # Handle empty response (e.g. think-only output)
        if not response or not response.strip():
            logger.warning("LLM returned empty response on iteration %d", iteration + 1)
            # Retry by asking more explicitly
            messages.append({"role": "assistant", "content": ""})
            messages.append({"role": "user", "content": "Please respond with a JSON tool call. For example: {\"tool\": \"search_trials_db\", \"args\": {\"query\": \"your search\"}}"})
            continue

        parsed = _try_parse_json(response)

        # LLM produced non-JSON — try to salvage or retry
        if parsed is None:
            logger.warning("LLM produced non-JSON: %s", response[:200])
            if all_tool_results:
                return _build_response(
                    query,
                    summary=response[:300],
                    results=all_tool_results,
                    sources=sources,
                    tool_calls_made=tool_calls_made,
                    start=start,
                )
            # On first iteration, give the LLM another chance
            if iteration == 0:
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": "You must respond with ONLY a JSON object. Choose a tool and call it. Example: {\"tool\": \"search_trials_db\", \"args\": {\"query\": \"CNS\", \"status\": \"TERMINATED\"}}"})
                continue
            return _error_response(query, "Agent produced invalid output", sources, 0, start)

        # Agent is done
        if parsed.get("done"):
            final_parsed = parsed
            break

        # Tool call
        tool_name = parsed.get("tool")
        tool_args = parsed.get("args", {})

        if not tool_name or tool_name not in TOOL_REGISTRY:
            available = list(TOOL_REGISTRY.keys())
            err_msg = f"Unknown tool '{tool_name}'. Available: {available}"
            logger.warning(err_msg)
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Error: {err_msg}"})
            continue

        # Execute tool
        logger.info("Calling tool: %s(%s)", tool_name, json.dumps(tool_args)[:100])
        try:
            result = TOOL_REGISTRY[tool_name]["execute"](**tool_args)
        except Exception as e:
            logger.error("Tool %s failed: %s", tool_name, e)
            result = [{"error": f"Tool execution failed: {e}"}]

        tool_calls_made += 1
        sources.add(SOURCE_MAP.get(tool_name, tool_name))
        all_tool_results.extend(result if isinstance(result, list) else [result])

        # Feed result back to LLM (truncated)
        result_str = json.dumps(result, default=str)
        if len(result_str) > 3000:
            result_str = result_str[:2997] + "..."

        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"Tool result for {tool_name}:\n{result_str}"})

    # Build final response
    if final_parsed:
        return _build_response(
            query,
            summary=final_parsed.get("summary", ""),
            results=final_parsed.get("results", all_tool_results),
            sources=sources,
            tool_calls_made=tool_calls_made,
            start=start,
        )

    # Loop exhausted without "done" — return what we have
    return _build_response(
        query,
        summary="Agent reached maximum iterations",
        results=all_tool_results,
        sources=sources,
        tool_calls_made=tool_calls_made,
        start=start,
    )


def _build_response(query, summary, results, sources, tool_calls_made, start):
    elapsed = round(time.time() - start, 2)
    logger.info("Agent finished: %d tool calls, %.2fs, sources=%s", tool_calls_made, elapsed, sources)
    return {
        "query": query,
        "summary": summary,
        "results": results,
        "sources": sorted(sources),
        "tool_calls_made": tool_calls_made,
        "elapsed_seconds": elapsed,
        "agent_model": llm.MODEL,
    }


def _error_response(query, error_msg, sources, tool_calls_made, start):
    elapsed = round(time.time() - start, 2)
    logger.error("Agent error: %s", error_msg)
    return {
        "query": query,
        "summary": "",
        "results": [],
        "sources": sorted(sources),
        "tool_calls_made": tool_calls_made,
        "elapsed_seconds": elapsed,
        "agent_model": llm.MODEL,
        "error": error_msg,
    }
