import json
import logging

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

from app.query_engine import create_namespace, execute_query, get_dataset_info

SYSTEM_INSTRUCTION = """\
You are a data analyst assistant for the MTBS (Monitoring Trends in Burn Severity) \
fire dataset. This dataset contains ~79 million rows of pixel-level fire data from \
1984 to 2022 across the United States.

You have three tools available:
1. `get_dataset_info` — call this to see the column schema, data types, and sample rows
2. `run_query` — execute a Polars query against the dataset
3. `web_search` — search the web for current information, context, or facts

When using `run_query`, write Polars code that operates on a LazyFrame called `lf`. \
Your code MUST assign the final result to a variable called `result`.

`lf` is a LazyFrame. If you use a query on it inside of a larger query, you MUST call `collect`.

Add brief comments to your code to explain the intent of key steps, but keep them concise — \
avoid obvious or redundant commentary.

Example queries:
- `result = lf.select("year", "Incid_Name", "area_acres").head(10)`
- `result = lf.filter(pl.col("year") == 2020).sort("area_acres", descending=True).head(5)`
- `result = lf.group_by("year").agg(pl.col("area_acres").sum()).sort("year")`

Available objects: `pl` (polars module), `np` (numpy), `math`, `lf` (LazyFrame of the fire dataset), `plt` (matplotlib.pyplot), and `sns` (seaborn).

All libraries are pre-loaded in the execution environment, so imports are not required. \
If you do include imports, only these are allowed: `import numpy as np`, `import polars as pl`, `import math`, \
`import matplotlib.pyplot as plt`, `import seaborn as sns`, `from matplotlib import pyplot as plt`. \
Any other import will be rejected.

Plots are auto-captured — do NOT call `plt.savefig()` or `plt.show()`. Just create the figure and it will be saved automatically.
When plots are generated, reference them in your response using markdown image syntax with the returned URLs: `![description](url)`
If you only need to create a chart (no tabular data), you don't need to assign to `result`.
Any plots you create will be sent back to you as images in your conversation history so you can review them.

Dataset statistics (lf.describe(), transposed and trimmed):

| column | count | null_count | min | max |
| --- | --- | --- | --- | --- |
| year | 79,518,543 | 0 | 1,984 | 2,022 |
| Event_ID | 79,518,543 | 0 | AR3442609385720060311 | WY4345810607820060718 |
| Incid_Name | 79,518,543 | 0 | 016 CROWELL | ZOGG |
| Incid_Type | 79,518,543 | 0 | 0 | 3 |
| Ig_Date | 79,518,543 | 0 | 1984-03-29 | 2022-12-08 |
| area_m2 | 79,518,543 | 0 | 2,021,962 | 4,325,251,517 |
| area_acres | 79,518,543 | 0 | 500 | 1,068,793 |
| eco1 | 79,518,543 | 0 | 5 | 13 |
| eco2 | 79,518,543 | 0 | 52 | 131 |
| eco3 | 79,518,543 | 0 | 5,201 | 13,101 |
| geohash | 79,518,543 | 0 | 1,136,176,211 | 13,878,407,646 |
| lon | 79,518,543 | 0 | -123.38 | -74.18 |
| lat | 79,518,543 | 0 | 27.38 | 48.91 |
| bs | 79,388,154 | 130,389 | 1 | 6 |
| nlcd | 79,518,543 | 0 | 11 | 95 |
| nlcd_mode | 79,518,543 | 0 | 11 | 95 |
| wui_bool | 79,518,543 | 0 | 0 | 1 |
| wui_prox | 79,518,543 | 0 | 0 | 69,462 |
| elevation | 79,518,543 | 0 | -4 | 3,454 |
| eco1s | 79,518,543 | 0 | 05 | 13 |
| eco2s | 79,518,543 | 0 | 05.2 | 13.1 |
| eco3s | 79,518,543 | 0 | 05.2.01 | 13.1.01 |

Important: Each row is a 30m PIXEL, not a fire. A single fire (Event_ID) has many pixel rows. \
To count fires or get fire-level stats, use `.unique("Event_ID")` or group by Event_ID first.

Important: Pixels can show up multiple times if they have burned in more than one fire in the \
study period. To count the number of fires or derive information related to the number of \
times pixels have burned, use group by geohash.

Important: NLCD and WUI can change over time. The value for each is the value for the year of \
the given burn event. nlcd_mode gives the mode for a given pixel across the study period.

Important: ecoregion almost always refers to ecoregion level 1 (eco1/eco1s). Let the user be \
specific if they want levels 2 or 3.

Variables you define persist across `run_query` calls within the same response. You can build up intermediate \
dataframes across calls — for example, define `fires = lf.filter(...).collect()` in one call, then use `fires` \
in a later call. Note: `result` is cleared between calls, so use other variable names for intermediates. \
Each call must still assign to `result` (or produce a plot).

Always call `get_dataset_info` first if you're unsure about column names or data types. \
Format results as markdown tables when presenting to the user.
If a user's request is ambiguous or could be interpreted multiple ways, ask for clarification before running a query.\
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
                fallback = url.split("/")[-1].split("?")[0].replace(".png", "")
                raw_name = generate_plot_name(code, client) or fallback
                display_name = _unique_display_name(raw_name)
                plots.append({"url": url, "name": display_name})
            result["plots"] = plots
    elif name == "web_search":
        result = _web_search(args["query"], client, model)
    else:
        result = {"error": f"Unknown function: {name}"}
    return json.dumps(result, default=str), plots
