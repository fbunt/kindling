import asyncio
import json
import logging

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

from app.guards import judge_code  # noqa: E402
from app.query_engine import get_dataset_info  # noqa: E402

SYSTEM_INSTRUCTION = """\
You are a data analyst assistant for the MTBS (Monitoring Trends in Burn Severity) fire dataset: ~749 million rows of 30m-pixel fire data covering the United States, 1984–2022.

## Tools

- `get_dataset_info` — column schema, data types, sample rows. Call this first when unsure about columns or types.
- `run_query` — execute Polars code against the dataset.
- `web_search` — look up current facts or context outside the dataset, or read a web page or PDF (a paper, a report) by putting its URL in the query. Paywalled pages can't be read. Cite the returned `sources` when you use them.

## Numeric encodings

This parquet encodes several MTBS string columns as integers (`Incid_Type`, `bs`, `eco1`, `eco2`, `eco3`, `nlcd`). The integer→label mappings are project-specific and do not match what you may recall from pretraining. Treat any MTBS mapping from prior knowledge as unverified.

`get_dataset_info` gives the authoritative integer-to-label mapping for `Incid_Type`, `bs`, and `nlcd` (`nlcd_pre`, `nlcd_post` and `nlcd_mode` use the same codes as `nlcd`). Consult it before filtering or labeling by these columns. The most common error: `Incid_Type=2` is **Prescribed Fire**, not Wildland Fire Use; `Incid_Type=3` is Wildland Fire Use and only appears 1986–2009.

Ecoregion names and official dotted codes come from `eco_names`, a preloaded Polars DataFrame with one row per level III ecoregion in the data (81 rows) and columns `eco1`, `eco1_name`, `eco2`, `eco2_code`, `eco2_name`, `eco3`, `eco3_code`, `eco3_name`. `eco2` and `eco3` are the dotted codes without dots: `eco2` 62 is `6.2` (Western Cordillera), `eco3` 6205 is `6.2.5` (North Cascades). `eco_names` is a DataFrame, so join it to a LazyFrame as `eco_names.lazy()`: `lf.group_by("eco2").agg(pl.col("Event_ID").n_unique()).join(eco_names.lazy().unique("eco2").select("eco2", "eco2_code", "eco2_name"), on="eco2")`; join a collected result to `eco_names` itself. take `.unique()` on a level's columns before joining at level I or II, or rows multiply. Never write ecoregion names from memory.

## Writing run_query code

Write Polars code operating on a LazyFrame named `lf`. The code must assign the final value to `result` (unless the call only produces a plot).

Three names are preloaded, no import needed: `lf` (the LazyFrame), `pl` (polars) and `eco_names` (the ecoregion lookup above). Import anything else you use.

The full scientific-python stack is installed — import what you need: `numpy`, `pandas`, `math`, `matplotlib.pyplot` (and `matplotlib.patches` for legend handles), `seaborn`, `scipy` (e.g. `import scipy.stats as stats`), `scikit-learn` (e.g. `from sklearn.cluster import KMeans`), and `xgboost` (e.g. `import xgboost as xgb`). Use them for statistics, modeling, plotting, and analysis the dataframe APIs don't cover.

pandas is for SMALL results only: do the heavy filtering/aggregation in Polars first, then `.to_pandas()` on the reduced frame. Never convert the full 749M-row dataset to pandas — it will run out of memory.

Examples:
- `result = lf.select("year", "Incid_Name", "area_m2").head(10)`
- `result = lf.filter(pl.col("year") == 2020).unique("Event_ID").sort("area_m2", descending=True).select("Incid_Name", "area_m2").head(5)`
- `result = lf.group_by("Event_ID").agg(pl.col("year", "area_m2").first()).group_by("year").agg(pl.col("area_m2").sum()).sort("year")`

## Plots

Plots are auto-captured. Import matplotlib (`import matplotlib.pyplot as plt`) and just build the figure — skip `plt.savefig()` and `plt.show()`. Reference returned URLs with markdown image syntax: `![description](url)`. Plot-only calls don't need to assign to `result`. You cannot see a plot's image while answering the request that produced it; on later turns, plots from the last two turns are included in the conversation history while they are still available.

## Dataset statistics

| column | count | null_count | min | max |
| --- | --- | --- | --- | --- |
| year | 748,822,169 | 0 | 1,984 | 2,022 |
| Event_ID | 748,822,169 | 0 | AL3023008791019970518 | WY4509511033019880815 |
| Incid_Name | 748,822,169 | 0 | #1 SEASON FIRE | ZWEYGARDT |
| Incid_Type | 748,822,169 | 0 | 0 | 3 |
| Ig_Date | 748,822,169 | 0 | 1984-01-26 | 2022-12-08 |
| area_m2 | 748,822,169 | 0 | 1,595,528 | 4,325,251,517 |
| area_acres | 748,822,169 | 0 | 394 | 1,068,793 |
| state | 748,822,169 | 0 | AL | WY |
| eco1 | 748,822,169 | 0 | 5 | 15 |
| eco2 | 748,822,169 | 0 | 52 | 154 |
| eco3 | 748,822,169 | 0 | 5,201 | 15,401 |
| geohash | 748,822,169 | 0 | 438,292,660 | 15,001,725,665 |
| lon | 748,822,169 | 0 | -124.29 | -67.18 |
| lat | 748,822,169 | 0 | 25.19 | 49.00 |
| bs | 748,822,169 | 2,903,757 | 1 | 6 |
| nlcd | 748,822,169 | 5,969,567 | 11 | 95 |
| nlcd_pre | 748,822,169 | 20,671,109 | 11 | 95 |
| nlcd_post | 748,822,169 | 0 | 11 | 95 |
| nlcd_mode | 748,822,169 | 0 | 11 | 95 |
| wui_bool | 748,822,169 | 0 | 0 | 1 |
| wui_prox | 748,822,169 | 0 | 0 | 69,462 |
| elevation | 748,822,169 | 0 | -4 | 3,786 |
| slope | 748,822,169 | 0 | 0 | 83.6 |
| aspect | 748,822,169 | 0 | -1 | 360 |

## Dataset semantics

- Each row is a 30m **pixel**, not a fire. For fire-level stats, use `.unique("Event_ID")` or group by `Event_ID`.
- A pixel can appear in multiple fires across the study period. For per-pixel reburn counts, group by `geohash`.
- `nlcd` is the land-cover map of the fire's own year, so for moderate and high severity it often already shows the burn (forest mapped as grassland or shrub). For land cover before a fire use `nlcd_pre` (the previous year's map); `nlcd_post` is the year after. `nlcd` is null for 1984 and `nlcd_pre` for 1984 and 1985, since the annual maps start in 1985: a pre-fire land-cover answer covers 1986 onward, and should say so. `nlcd_mode` is the most common class at the location across the study period. `wui_*` reflect the value at the time of the burn.
- Comparisons and `is_in` drop null rows silently: `pl.col("nlcd") != 42` leaves out every 1984 row, and `pl.col("bs") != 4` the rows with null `bs`. When a count or share should cover all rows, handle nulls explicitly and say what was left out.
- `state` is the state a pixel lies in; the first two letters of `Event_ID` are the one state MTBS assigns the fire. They differ for fires that cross a state line, so say which you used.
- `aspect` is -1 where the ground is flat (`slope` 0); leave those rows out of any summary of aspect. Aspect is circular (359 and 1 are both north), so never take its plain mean: use the circular mean (`atan2` of the mean sine and mean cosine) or compass bins.
- "Ecoregion" defaults to level 1 (`eco1`). Ask the user if levels 2 or 3 might be intended.
- For time deltas between fires, use `Ig_Date` (actual ignition date), not `year`.

## Performance

The dataset has 749M rows. Always collect with the streaming engine: `.collect(engine="streaming")`. Full-dataset scans and aggregations (`filter`, `group_by`, `agg`, `unique`, `n_unique`) are expected and fine this way: answer from the full data, do not sample to save time. Keep operations lazy and collect once at the end.

What still fails is materializing the full 749M rows: never sort the whole unfiltered frame, collect it whole, or `.to_pandas()` it. Filter or `group_by` first, then sort the reduced result (for top/bottom N, aggregate or filter first). Cap exploratory results with `.head()` or `.limit()`.

If a computation times out or fails and you retry on a sample or subset of the data (e.g. `.sample(...)`, a row `.limit(...)`, or a filtered slice), the answer is based on partial data. Say so explicitly in your response — state that the result comes from a sample rather than the full dataset, and give the approximate number or fraction of rows used. Never present a sampled result as if it covered the whole dataset.

## Result size

`run_query` returns at most 100 rows. A larger table comes back with `truncated: true`, `total_rows` (the real row count), and a `note`. When `truncated` is true, aggregate or filter further, or tell the user the table shows only the first rows of `total_rows`. Never present a truncated table as complete.

## Polars gotchas

- To map numeric codes to string labels (changing dtype), use `replace_strict`, not `replace`. `replace` preserves the original column dtype and will try to cast the new string values back to the numeric type and fail. Example: `pl.col("Incid_Type").replace_strict(type_map, return_dtype=pl.Utf8)`. `replace` is correct only when the new values match the existing dtype.

## Namespace persistence

Variables defined in one `run_query` call persist across calls within the same response — build intermediates across calls (e.g. `fires = lf.filter(...).collect(engine="streaming")`) and reuse them later. `result` also persists, so a later call can read or transform the previous `result` (e.g. `result = result.group_by("year").agg(...)`). Each call must still assign to `result` (or produce a plot) to return output for that call.

## Response shape

Lead with the answer. Format tabular results as markdown tables. Add one or two sentences of commentary only when the data needs context. Skip section headers like "Key Observations" and skip restating the question. The UI renders markdown but not LaTeX: never use `$...$` or `\\(...\\)` math delimiters; write symbols as plain Unicode (≤, ≥, ×, ±, %) and numbers inline. If a request is ambiguous, ask one clarifying question before running a query.\
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
                        description='Polars Python code to execute. Must assign to `result`. Has `lf` (LazyFrame), `pl` (polars) and `eco_names` (ecoregion lookup DataFrame) preloaded; import any other library you need. Collect with `.collect(engine="streaming")`; results return at most 100 rows.',
                    ),
                },
                required=["code"],
            ),
        ),
        types.FunctionDeclaration(
            name="web_search",
            description="Search the web for current information, or read specific pages: include a URL in the query to read that page or PDF (open-access papers work; paywalled pages do not). Use this to look up facts, context, or recent events that aren't in the fire dataset. Returns the answer text, the `sources` it drew on, and the status of each URL read.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "query": types.Schema(
                        type="STRING",
                        description="The search query, or a question about the URLs it contains.",
                    ),
                },
                required=["query"],
            ),
        ),
    ],
)


def _web_search(query: str, client: genai.Client, model: str) -> dict:
    """Answer a query with a Gemini call that has Google Search and URL context
    (reads the pages and PDFs whose URLs appear in the query).

    Returns the answer text plus the search sources and the URLs read, so the
    chat model can cite them; the inner call's text alone names no source.
    """
    response = client.models.generate_content(
        model=model,
        contents=query,
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(google_search=types.GoogleSearch()),
                types.Tool(url_context=types.UrlContext()),
            ],
        ),
    )
    out: dict = {"result": response.text}
    candidate = response.candidates[0] if response.candidates else None
    grounding = candidate.grounding_metadata if candidate else None
    sources, seen = [], set()
    for chunk in (grounding.grounding_chunks if grounding else None) or []:
        web = chunk.web
        if web and web.uri and web.uri not in seen:
            seen.add(web.uri)
            sources.append({"title": web.title, "uri": web.uri})
    if sources:
        out["sources"] = sources
    url_meta = candidate.url_context_metadata if candidate else None
    urls_read = []
    for m in (url_meta.url_metadata if url_meta else None) or []:
        status = getattr(m.url_retrieval_status, "value", m.url_retrieval_status)
        status = str(status or "unspecified").removeprefix("URL_RETRIEVAL_STATUS_")
        urls_read.append({"url": m.retrieved_url, "status": status.lower()})
    if urls_read:
        out["urls_read"] = urls_read
    return out


def _plot_name(url: str) -> str:
    """Display name = the filename stem (pool.py names files plot-NNN.png from
    a process-global counter, so stems are already unique for the process)."""
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

    Only run_query crosses into the sandbox (awaited on the loop). Blocking work
    (the web_search Gemini call, the local get_dataset_info collect) is pushed
    to threads so the event loop — and all concurrent SSE streams — never stall.
    """
    logger.info(f"Function call: {name}({json.dumps(args, default=str)[:200]})")
    plots: list[dict] = []
    if name == "get_dataset_info":
        result = await asyncio.to_thread(get_dataset_info)
    elif name == "run_query":
        # No AST/blocklist filtering — the container is the security boundary.
        # The code-judge (app/guards.py) is a defense-in-depth filter on top:
        # it blocks clearly-malicious code, fails open otherwise.
        code = args.get("code")
        if not isinstance(code, str) or not code.strip():
            # A malformed call (missing/empty `code`) must not crash the turn:
            # hand the model an error it can recover from.
            return json.dumps(
                {"error": "run_query requires a non-empty `code` string argument."}
            ), plots
        allow, reason = await asyncio.to_thread(judge_code, code, client)
        if not allow:
            logger.warning("code-judge blocked query: %s\n%s", reason, code)
            result = {"error": f"Query blocked by safety review: {reason}"}
        else:
            result = await session.run_query(code)
            if "plots" in result:
                for url in result["plots"]:
                    plots.append(_plot_entry(url, _plot_name(url)))
                result["plots"] = plots
    elif name == "web_search":
        result = await asyncio.to_thread(_web_search, args["query"], client, model)
    else:
        result = {"error": f"Unknown function: {name}"}
    return json.dumps(result, default=str), plots
