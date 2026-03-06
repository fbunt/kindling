import json
import logging

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

from app.query_engine import get_dataset_info, execute_query

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

Example queries:
- `result = lf.select("year", "Incid_Name", "area_acres").head(10)`
- `result = lf.filter(pl.col("year") == 2020).sort("area_acres", descending=True).head(5)`
- `result = lf.group_by("year").agg(pl.col("area_acres").sum()).sort("year")`

Available objects: `pl` (polars module), `lf` (LazyFrame of the fire dataset), `plt` (matplotlib.pyplot), and `sns` (seaborn).

IMPORTANT: Do NOT use `import` statements — all libraries are pre-loaded in the execution environment. \
Use `plt`, `sns`, `pl`, and `lf` directly.

Plots are auto-captured — do NOT call `plt.savefig()` or `plt.show()`. Just create the figure and it will be saved automatically.
When plots are generated, reference them in your response using markdown image syntax with the returned URLs: `![description](url)`
If you only need to create a chart (no tabular data), you don't need to assign to `result`.

Key columns:
- year (UInt16): Fire year (1984-2022)
- Incid_Name (String): Fire incident name
- Event_ID (String): Unique fire event identifier
- area_acres (Float64): Fire area in acres
- bs (UInt8): Burn severity class (1=Unburned, 2=Low, 3=Moderate, 4=High, 5=Increased Greenness, 6=Non-processing)
- Ig_Date (Datetime): Ignition date
- lat/lon (Float64): Latitude/longitude
- elevation (Float32): Elevation
- eco1s/eco2s/eco3s (String): Ecoregion levels
- nlcd (UInt8): NLCD land cover class
- wui_flag (UInt8): Wildland-Urban Interface flag

Important: Each row is a 30m PIXEL, not a fire. A single fire (Event_ID) has many pixel rows. \
To count fires or get fire-level stats, use `.unique("Event_ID")` or group by Event_ID first.

Always call `get_dataset_info` first if you're unsure about column names or data types. \
Format results as markdown tables when presenting to the user.\
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
            description="Execute a Polars query against the MTBS fire dataset. The code must use the LazyFrame `lf` and polars `pl`, and assign the result to a variable called `result`.",
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


def execute_function_call(
    name: str, args: dict, client: genai.Client, model: str
) -> tuple[str, list[dict]]:
    """Dispatch a function call from Gemini. Returns (json_result, plots_list)."""
    plots = []
    if name == "get_dataset_info":
        result = get_dataset_info()
    elif name == "run_query":
        result = execute_query(args["code"])
        # Generate display names for any plots
        if "plots" in result:
            code = args.get("code", "")
            for url in result["plots"]:
                display_name = generate_plot_name(code, client) or url.split("/")[-1].replace(".png", "")
                plots.append({"url": url, "name": display_name})
            result["plots"] = plots
    elif name == "web_search":
        result = _web_search(args["query"], client, model)
    else:
        result = {"error": f"Unknown function: {name}"}
    return json.dumps(result, default=str), plots
