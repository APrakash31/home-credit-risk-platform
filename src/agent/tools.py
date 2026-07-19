import json
import pandas as pd
import duckdb
from pathlib import Path

from src.explain.reason_codes import ApplicantExplainer
from src.agent.retrieval import search_knowledge_base

ROOT = Path(__file__).resolve().parents[2]
_full = ROOT / "data" / "processed" / "features.parquet"
_demo = ROOT / "app" / "demo_data" / "features_sample.parquet"
FEATURES = _full if _full.exists() else _demo

_explainer = None
_df = None


def _get_explainer():
    global _explainer
    if _explainer is None:
        _explainer = ApplicantExplainer()
    return _explainer


def _get_df():
    global _df
    if _df is None:
        _df = pd.read_parquet(FEATURES)
    return _df


def score_applicant(applicant_id: int) -> str:
    """Score one applicant and explain the drivers via SHAP."""
    df = _get_df()
    row = df[df["SK_ID_CURR"] == applicant_id]
    if row.empty:
        return json.dumps({"error": f"Applicant {applicant_id} not found"})

    X = row.drop(columns=[c for c in ["TARGET", "SK_ID_CURR"] if c in row.columns])
    X.columns = [c.replace(" ", "_").replace(":", "_").replace(",", "_") for c in X.columns]

    result = _get_explainer().explain(X)
    result["applicant_id"] = int(applicant_id)
    return json.dumps(result, indent=2)


def query_portfolio(sql: str) -> str:
    """Run a read-only SQL query against the applicant feature table."""
    lowered = sql.lower().strip()
    forbidden = ["insert", "update", "delete", "drop", "alter", "create", "attach", "copy"]
    if any(word in lowered for word in forbidden):
        return json.dumps({"error": "Only read-only SELECT queries are permitted."})
    if not lowered.startswith("select"):
        return json.dumps({"error": "Query must begin with SELECT."})

    try:
        con = duckdb.connect()
        con.execute(f"CREATE VIEW applicants AS SELECT * FROM read_parquet('{FEATURES}')")
        result = con.execute(sql).fetchdf().head(50)
        return result.to_json(orient="records", indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def lookup_policy(question: str) -> str:
    """Retrieve credit policy and feature definitions."""
    return search_knowledge_base(question, top=3)


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "score_applicant",
            "description": (
                "Score a specific loan applicant and return their default probability, "
                "risk band, and the SHAP-derived factors driving the decision with "
                "adverse action reason codes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "applicant_id": {
                        "type": "integer",
                        "description": "The SK_ID_CURR identifier of the applicant",
                    }
                },
                "required": ["applicant_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_portfolio",
            "description": (
                "Run a read-only SQL SELECT query against the applicant table, which is "
                "named 'applicants'. Use for aggregate questions about the loan book, "
                "such as default rates by segment. Columns include TARGET (1 = default), "
                "SK_ID_CURR, EXT_SOURCE_1/2/3, INST_IS_LATE_MEAN, ANNUITY_INCOME_RATIO, "
                "CREDIT_INCOME_RATIO, YEARS_EMPLOYED, YEARS_BIRTH, BUREAU_ACTIVE_COUNT."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "A read-only SELECT query"}
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_policy",
            "description": (
                "Search the credit policy, data dictionary, and adverse action reason "
                "codes. Use whenever a question concerns lending rules, thresholds, "
                "escalation requirements, or what a model feature means."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The policy or definition question"}
                },
                "required": ["question"],
            },
        },
    },
]

TOOL_REGISTRY = {
    "score_applicant": score_applicant,
    "query_portfolio": query_portfolio,
    "lookup_policy": lookup_policy,
}