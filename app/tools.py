import asyncio
import json
import logging

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

from app.query_engine import execute_query, get_dataset_info  # noqa: E402

SYSTEM_INSTRUCTION = """\
You are a data analyst assistant for the MTBS (Monitoring Trends in Burn Severity) fire dataset: ~745 million rows of 30m-pixel fire data covering the United States, 1984–2022.

## Tools

- `get_dataset_info` — column schema, data types, sample rows. Call this first when unsure about columns or types.
- `run_query` — execute Polars code against the dataset.
- `web_search` — look up current facts or context outside the dataset.

## Numeric encodings

This parquet encodes several MTBS string columns as integers (`Incid_Type`, `bs`, `eco1`, `eco2`, `eco3`, `nlcd`). The integer→label mappings are project-specific and do not match what you may recall from pretraining. Treat any MTBS mapping from prior knowledge as unverified.

Before filtering or labeling by these columns, consult `get_dataset_info` for the authoritative mapping. The most common error: `Incid_Type=2` is **Prescribed Fire**, not Wildland Fire Use; `Incid_Type=3` is Wildland Fire Use and only appears 1988–2009.

## Writing run_query code

Write Polars code operating on a LazyFrame named `lf`. The code must assign the final value to `result` (unless the call only produces a plot).

Available objects (pre-loaded; imports unnecessary): `pl`, `np`, `math`, `lf`, `plt`, `sns`, `Patch` (matplotlib.patches.Patch, for legend handles).

If you do include imports, only these are accepted: `import numpy as np`, `import polars as pl`, `import math`, `import matplotlib.pyplot as plt`, `import seaborn as sns`, `from matplotlib import pyplot as plt`. Anything else is rejected.

Examples:
- `result = lf.select("year", "Incid_Name", "area_m2").head(10)`
- `result = lf.filter(pl.col("year") == 2020).sort("area_m2", descending=True).head(5)`
- `result = lf.group_by("year").agg(pl.col("area_m2").sum()).sort("year")`

## Plots

Plots are auto-captured. Just build the figure — skip `plt.savefig()` and `plt.show()`. Reference returned URLs with markdown image syntax: `![description](url)`. Plot-only calls don't need to assign to `result`. Generated plot images are returned to you in conversation history for review.

## Dataset statistics

| column | count | null_count | min | max |
| --- | --- | --- | --- | --- |
| year | 745,294,556 | 0 | 1,984 | 2,022 |
| Incid_Name | 745,294,556 | 0 | #1 SEASON FIRE | ZWEYGARDT |
| Event_ID | 745,294,556 | 0 | AL3023008791019970518 | WY4509511033019880815 |
| Incid_Type | 745,294,556 | 0 | 0 | 3 |
| area_m2 | 745,294,556 | 0 | 1,595,528 | 4,325,251,517 |
| geohash | 745,294,556 | 0 | 438,292,660 | 15,001,725,665 |
| bs | 745,294,556 | 2,903,607 | 1 | 6 |
| Ig_Date | 745,294,556 | 0 | 1984-01-26 | 2022-12-08 |
| lat | 745,294,556 | 0 | 25.19 | 49.00 |
| lon | 745,294,556 | 0 | -124.29 | -67.18 |
| elevation | 745,294,556 | 0 | -4 | 3,786 |
| eco1 | 745,294,556 | 0 | 5 | 15 |
| eco2 | 745,294,556 | 0 | 52 | 154 |
| eco3 | 745,294,556 | 0 | 5,201 | 15,401 |
| eco1s | 745,294,556 | 0 | 05 | 15 |
| eco2s | 745,294,556 | 0 | 05.2 | 15.4 |
| eco3s | 745,294,556 | 0 | 05.2.01 | 15.4.01 |
| nlcd | 745,294,556 | 0 | 11 | 95 |
| nlcd_mode | 745,294,556 | 0 | 11 | 95 |
| wui_bool | 745,294,556 | 0 | 0 | 1 |
| wui_prox | 745,294,556 | 0 | 0 | 69,462 |

## Dataset semantics

- Each row is a 30m **pixel**, not a fire. For fire-level stats, use `.unique("Event_ID")` or group by `Event_ID`.
- A pixel can appear in multiple fires across the study period. For per-pixel reburn counts, group by `geohash`.
- `nlcd` and `wui_*` reflect the value at the time of that pixel's burn event; `nlcd_mode` is the mode across the study period.
- "Ecoregion" defaults to level 1 (`eco1`/`eco1s`). Ask the user if levels 2 or 3 might be intended.
- For time deltas between fires, use `Ig_Date` (actual ignition date), not `year`.

## Performance

The dataset has 745M rows. Filter or `group_by` before sorting. Cap exploratory results with `.head()` or `.limit()`. Keep operations lazy (`filter`, `group_by`, `agg`) and `.collect()` once at the end. For top/bottom N, aggregate or filter first, then sort the reduced result. Full-dataset scans or sorts will time out.

## Polars gotchas

- To map numeric codes to string labels (changing dtype), use `replace_strict`, not `replace`. `replace` preserves the original column dtype and will try to cast the new string values back to the numeric type and fail. Example: `pl.col("eco1").replace_strict(eco1_map, return_dtype=pl.Utf8)`. `replace` is correct only when the new values match the existing dtype.

## Namespace persistence

Variables defined in one `run_query` call persist across calls within the same response — build intermediates across calls (e.g. `fires = lf.filter(...).collect()`) and reuse them later. `result` also persists, so a later call can read or transform the previous `result` (e.g. `result = result.group_by("year").agg(...)`). Each call must still assign to `result` (or produce a plot) to return output for that call.

## Response shape

Lead with the answer. Format tabular results as markdown tables. Add one or two sentences of commentary only when the data needs context. Skip section headers like "Key Observations" and skip restating the question. If a request is ambiguous, ask one clarifying question before running a query.\
"""

FIRE_DATA_TOOLS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_dataset_info",
            description="Get the dataset schema, approximate row count, and sample rows. Call this to understand the available columns and data types before writing queries.",
            parameters=types.Schema(
                type="OBJECT",
                properties={},
            ),
        ),
        types.FunctionDeclaration(
            name="run_query",
            description="Execute a Polars query against the MTBS fire dataset. Variables from previous run_query calls in the same turn are available. The code must assign the result to a variable called `result`.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "code": types.Schema(
                        type="STRING",
                        description="Polars Python code to execute. Must assign to `result`. Has access to `lf` (LazyFrame) and `pl` (polars).",
                    ),
                },
                required=["code"],
            ),
        ),
        types.FunctionDeclaration(
            name="web_search",
            description="Search the web for current information. Use this to look up facts, context, or recent events that aren't in the fire dataset.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "query": types.Schema(
                        type="STRING",
                        description="The search query.",
                    ),
                },
                required=["query"],
            ),
        ),
    ],
)


def generate_plot_name(code: str, client: genai.Client) -> str | None:
    """Call flash-lite to generate a short kebab-case name for a plot."""
    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=f"Generate a short kebab-case filename (no extension, max 5 words) for a plot created by this code. Reply with ONLY the filename, nothing else:\n\n{code}",
        )
        name = response.text.strip().strip("`").strip()
        # Sanitize: keep only lowercase alphanumeric and hyphens
        name = "-".join(w for w in name.lower().split("-") if w.isalnum())
        return name if name else None
    except Exception as e:
        logger.warning(f"generate_plot_name failed: {e}")
        return None


def _web_search(query: str, client: genai.Client, model: str) -> dict:
    """Perform a web search by calling Gemini with Google Search enabled."""
    response = client.models.generate_content(
        model=model,
        contents=query,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )
    return {"result": response.text}


_used_display_names: set[str] = set()


def _unique_display_name(name: str) -> str:
    """Append a numeric suffix if the display name has already been used."""
    if name not in _used_display_names:
        _used_display_names.add(name)
        return name
    n = 1
    while f"{name}-{n:03d}" in _used_display_names:
        n += 1
    unique = f"{name}-{n:03d}"
    _used_display_names.add(unique)
    return unique


def execute_function_call(
    name: str,
    args: dict,
    client: genai.Client,
    model: str,
    namespace: dict | None = None,
) -> tuple[str, list[dict]]:
    """Dispatch a function call from Gemini. Returns (json_result, plots_list)."""
    logger.info(f"Function call: {name}({json.dumps(args, default=str)[:200]})")
    plots = []
    if name == "get_dataset_info":
        result = get_dataset_info()
    elif name == "run_query":
        result = execute_query(args["code"], namespace)
        # Generate display names for any plots
        if "plots" in result:
            code = args.get("code", "")
            for url in result["plots"]:
                raw_name = generate_plot_name(code, client) or _plot_fallback(url)
                plots.append(_plot_entry(url, _unique_display_name(raw_name)))
            result["plots"] = plots
    elif name == "web_search":
        result = _web_search(args["query"], client, model)
    else:
        result = {"error": f"Unknown function: {name}"}
    return json.dumps(result, default=str), plots


def _plot_fallback(url: str) -> str:
    """Derive a fallback display name from a plot URL."""
    return url.split("/")[-1].split("?")[0].replace(".png", "")


def _plot_entry(url: str, display_name: str) -> dict:
    """Build a plot entry carrying both the cache-busted URL (for the frontend)
    and a clean on-disk path (for host-side reads — the URL's ?t= query string
    makes Path(url) point at a nonexistent file)."""
    return {"url": url, "name": display_name, "path": url.split("?")[0].lstrip("/")}


async def execute_function_call_async(
    name: str,
    args: dict,
    client: genai.Client,
    model: str,
    session,
) -> tuple[str, list[dict]]:
    """Async dispatch for the container sandbox path.

    Only run_query crosses into the sandbox (awaited on the loop). Every blocking
    Gemini call (generate_plot_name, web_search) and the local get_dataset_info
    collect are pushed to threads so the event loop — and all concurrent SSE
    streams — never stall.
    """
    logger.info(f"Function call: {name}({json.dumps(args, default=str)[:200]})")
    plots: list[dict] = []
    if name == "get_dataset_info":
        result = await asyncio.to_thread(get_dataset_info)
    elif name == "run_query":
        result = await session.run_query(args["code"])
        if "plots" in result:
            code = args.get("code", "")
            for url in result["plots"]:
                raw_name = (
                    await asyncio.to_thread(generate_plot_name, code, client)
                ) or _plot_fallback(url)
                plots.append(_plot_entry(url, _unique_display_name(raw_name)))
            result["plots"] = plots
    elif name == "web_search":
        result = await asyncio.to_thread(_web_search, args["query"], client, model)
    else:
        result = {"error": f"Unknown function: {name}"}
    return json.dumps(result, default=str), plots
