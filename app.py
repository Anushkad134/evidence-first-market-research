import os
import json
import re
import operator
import textwrap
from typing import TypedDict, Annotated

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# -------------------- Page setup and styling --------------------
st.set_page_config(
    page_title="Market Intelligence Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

st.markdown("""
<style>
    /* ---------- Base page ---------- */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: #f4f7fb !important;
    }
    [data-testid="stHeader"] {
        background: #f4f7fb !important;
    }
    [data-testid="stHeader"] * { color: #18263c !important; }

    /* ---------- Main-area text: always dark and readable ---------- */
    [data-testid="stMain"] .stMarkdown,
    [data-testid="stMain"] .stMarkdown p,
    [data-testid="stMain"] .stMarkdown li,
    [data-testid="stMain"] .stMarkdown td,
    [data-testid="stMain"] .stMarkdown th,
    [data-testid="stMain"] .stMarkdown h1,
    [data-testid="stMain"] .stMarkdown h2,
    [data-testid="stMain"] .stMarkdown h3,
    [data-testid="stMain"] .stMarkdown h4,
    [data-testid="stMain"] .stMarkdown strong,
    [data-testid="stMain"] label,
    [data-testid="stMain"] [data-testid="stWidgetLabel"] p,
    [data-testid="stMain"] [data-testid="stCaptionContainer"] {
        color: #18263c !important;
    }
    [data-testid="stMain"] .stMarkdown a { color: #0b6fa4 !important; }
    [data-testid="stMain"] .stMarkdown code {
        background: #e8eff8 !important; color: #10213a !important;
    }

    /* ---------- Markdown tables (competitor comparison) ---------- */
    [data-testid="stMain"] .stMarkdown table {
        border-collapse: collapse; width: 100%;
        background: white; border-radius: 10px; overflow: hidden;
    }
    [data-testid="stMain"] .stMarkdown th {
        background: #e8eff8 !important; border: 1px solid #d5e0ee !important;
    }
    [data-testid="stMain"] .stMarkdown td {
        border: 1px solid #e3eaf4 !important;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] { background: #101b31 !important; }
    [data-testid="stSidebar"] * { color: #eef5ff !important; }

    /* ---------- Hero banner (must stay white text) ---------- */
    [data-testid="stMain"] .hero {
        padding: 1.5rem 1.7rem; border-radius: 18px;
        background: linear-gradient(120deg, #101b31 0%, #173b67 65%, #087eaa 100%);
        margin-bottom: 1rem;
        box-shadow: 0 10px 28px rgba(16, 27, 49, .16);
    }
    [data-testid="stMain"] .hero h1 {
        color: #ffffff !important; margin: 0; font-size: 2.25rem;
    }
    [data-testid="stMain"] .hero p {
        color: #d9eaff !important; margin: .45rem 0 0 0; font-size: 1rem;
    }
    .small-note { color: #52637a; font-size: .9rem; }

    /* ---------- Metric cards ---------- */
    div[data-testid="stMetric"] {
        background: white !important; padding: 1rem; border-radius: 14px;
        border: 1px solid #e3eaf4;
        box-shadow: 0 2px 8px rgba(16, 27, 49, .05);
    }
    div[data-testid="stMetric"] * { color: #18263c !important; }

    /* ---------- Inputs ---------- */
    [data-testid="stMain"] input,
    [data-testid="stMain"] textarea,
    [data-testid="stMain"] [data-baseweb="select"] > div {
        background: #ffffff !important;
        color: #18263c !important;
        border-color: #cfdbea !important;
    }
    [data-testid="stMain"] [data-testid="stForm"] {
        background: #ffffff; border: 1px solid #e3eaf4; border-radius: 14px;
    }

    /* ---------- Tabs ---------- */
    div[data-testid="stTabs"] button { font-weight: 650; color: #52637a !important; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: #087eaa !important; }

    /* ---------- Expander ---------- */
    [data-testid="stExpander"] {
        background: #ffffff; border: 1px solid #e3eaf4; border-radius: 12px;
    }
    [data-testid="stExpander"] * { color: #18263c !important; }

    /* ---------- Buttons ---------- */
    [data-testid="stMain"] .stButton button[kind="primary"],
    [data-testid="stMain"] .stFormSubmitButton button[kind="primary"] {
        background: #087eaa !important; color: #ffffff !important; border: none;
    }
    [data-testid="stMain"] .stDownloadButton button {
        background: #ffffff !important; color: #18263c !important;
        border: 1px solid #cfdbea !important;
    }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.75)",
    font=dict(family="Inter, Arial, sans-serif", color="#18263c"),
    margin=dict(l=25, r=25, t=65, b=25),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hoverlabel=dict(bgcolor="#10213a", font_color="white"),
)

# -------------------- LLM and graph state --------------------
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0) if OPENAI_API_KEY else None
tavily = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

class ResearchState(TypedDict, total=False):
    company: str
    topic: str
    queries: list[str]
    sources: Annotated[list[dict], operator.add]
    competitor_analysis: str
    trends_analysis: str
    financial_analysis: str
    report: str

def ask_llm(instructions: str, context: str) -> str:
    if llm is None:
        raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")
    response = llm.invoke([
        SystemMessage(content=instructions),
        HumanMessage(content=context),
    ])
    return response.content if isinstance(response.content, str) else str(response.content)

def plan_research(state: ResearchState) -> dict:
    prompt = f"""
Create exactly five web search queries for a market-intelligence report.
Company/topic: {state['company']}
Research focus: {state['topic']}
Queries must cover: company overview and market position; competitors; industry trends;
recent news; financials or funding where relevant.
Return ONLY a JSON array of five strings. No markdown.
"""
    try:
        raw = ask_llm("You are a careful market research planner. Return valid JSON only.", prompt)
        match = re.search(r"\[[\s\S]*\]", raw)
        queries = json.loads(match.group(0)) if match else []
        queries = [str(q).strip() for q in queries if str(q).strip()][:5]
    except Exception:
        queries = []
    if len(queries) < 5:
        company = state["company"]
        focus = state["topic"]
        defaults = [
            f"{company} company overview market position {focus}",
            f"{company} main competitors comparison {focus}",
            f"{focus} industry trends and opportunities 2025 2026",
            f"{company} latest news product strategy partnerships",
            f"{company} revenue financial results funding annual report",
        ]
        queries = (queries + defaults)[:5]
    return {"queries": queries}

def web_research(state: ResearchState) -> dict:
    if tavily is None:
        raise ValueError("TAVILY_API_KEY is missing. Add it to your .env file.")
    all_sources = []
    seen = set()
    for query in state["queries"]:
        response = tavily.search(query=query, max_results=5, search_depth="basic")
        for item in response.get("results", []):
            url = item.get("url", "")
            if not url or url in seen:
                continue
            seen.add(url)
            all_sources.append({
                "query": query,
                "title": item.get("title", "Untitled"),
                "content": item.get("content", ""),
                "url": url,
                "score": item.get("score"),
            })
    return {"sources": all_sources}

def research_context(state: ResearchState) -> str:
    chunks = []
    for i, source in enumerate(state.get("sources", []), 1):
        chunks.append(
            f"SOURCE {i}\nQuery: {source.get('query')}\nTitle: {source.get('title')}\n"
            f"Content: {source.get('content')}\nURL: {source.get('url')}"
        )
    return "\n\n".join(chunks)[:42000]

def analyze_competitors(state: ResearchState) -> dict:
    prompt = f"""
Research target: {state['company']}
Focus: {state['topic']}
Using only the evidence below, write a competitor landscape in Markdown.
Include: direct/indirect competitors when supported, positioning, differentiators,
and a concise comparison table. Separate confirmed facts from qualitative judgments.
Cite claims with source URLs in Markdown. Never invent market shares, scores, prices,
or financial figures. If evidence is insufficient, say so explicitly.
EVIDENCE:
{research_context(state)}
"""
    return {"competitor_analysis": ask_llm("You are a skeptical market analyst. Ground every factual claim in evidence.", prompt)}

def analyze_trends(state: ResearchState) -> dict:
    prompt = f"""
Research target: {state['company']}
Focus: {state['topic']}
Write a concise, evidence-led market trends analysis in Markdown.
Cover emerging trends, customer/technology shifts, opportunities, risks, and implications.
Include dates where the sources provide them and cite source URLs.
Do not manufacture time series, forecasts, market sizes, or numerical scores.
Clearly label uncertain or conflicting evidence.
EVIDENCE:
{research_context(state)}
"""
    return {"trends_analysis": ask_llm("You are an evidence-driven industry research analyst.", prompt)}

def financial_snapshot(state: ResearchState) -> dict:
    prompt = f"""
Research target: {state['company']}
Focus: {state['topic']}
Write a financial and business snapshot in Markdown.
Report revenue, profit, valuation, funding, growth, or segment data only if present in sources.
For each figure, state the period and currency when available and cite its source URL.
Do not infer missing values or present estimates as confirmed facts. If data is unavailable,
explain what was and was not found.
EVIDENCE:
{research_context(state)}
"""
    return {"financial_analysis": ask_llm("You are a careful financial research analyst, not a financial adviser.", prompt)}

def synthesize_report(state: ResearchState) -> dict:
    prompt = f"""
Create an executive-ready market intelligence report for {state['company']} focused on {state['topic']}.
Use the following researched sections. Structure: executive summary, market/company position,
key findings, opportunities, risks, recommended next steps, evidence limitations.
Cite sources using URLs already included. Do not introduce new facts or unsupported numbers.
Keep it readable, specific, and decision-oriented.

COMPETITORS:
{state.get('competitor_analysis', '')}

TRENDS:
{state.get('trends_analysis', '')}

FINANCIAL SNAPSHOT:
{state.get('financial_analysis', '')}
"""
    return {"report": ask_llm("You synthesize reliable research into clear business intelligence. Never fabricate facts.", prompt)}

@st.cache_resource(show_spinner=False)
def build_graph():
    builder = StateGraph(ResearchState)
    builder.add_node("plan_research", plan_research)
    builder.add_node("web_research", web_research)
    builder.add_node("analyze_competitors", analyze_competitors)
    builder.add_node("analyze_trends", analyze_trends)
    builder.add_node("financial_snapshot", financial_snapshot)
    builder.add_node("synthesize_report", synthesize_report)
    builder.add_edge(START, "plan_research")
    builder.add_edge("plan_research", "web_research")
    builder.add_edge("web_research", "analyze_competitors")
    builder.add_edge("web_research", "analyze_trends")
    builder.add_edge("web_research", "financial_snapshot")
    builder.add_edge(["analyze_competitors", "analyze_trends", "financial_snapshot"], "synthesize_report")
    builder.add_edge("synthesize_report", END)
    return builder.compile()

# -------------------- Chart data extraction --------------------
def extract_chart_data(company: str, topic: str, sources: list[dict]) -> dict:
    """Ask for a small, sourced set of structured competitor attributes. Estimates are explicitly qualitative."""
    evidence = "\n\n".join(
        f"TITLE: {s['title']}\nTEXT: {s['content']}\nURL: {s['url']}"
        for s in sources[:18]
    )[:24000]
    prompt = f"""
Target: {company}; focus: {topic}
From the evidence, identify up to 8 relevant companies/organizations including the target if possible.
Return ONLY valid JSON with this schema:
{{
 "competitors":[
   {{"name":"name","market_presence":1-10,"innovation":1-10,"offering_breadth":1-10,
     "evidence_note":"short rationale","source_url":"URL from evidence"}}
 ],
 "financials":[
   {{"company":"name","metric":"revenue or funding or profit","period":"period",
     "value":number,"unit":"USD millions, USD billions, etc.","source_url":"URL"}}
 ]
}}
Important:
- market_presence, innovation and offering_breadth are qualitative analyst ratings inferred from descriptions, NOT verified quantitative metrics. Use integers 1–10 and give a rationale.
- financials must contain ONLY exact numerical figures clearly stated in the evidence with their period and unit. Do not convert currencies or invent values.
- If data is not available, return an empty array.
EVIDENCE:
{evidence}
"""
    raw = ask_llm("Return strict JSON only. Never fabricate data.", prompt)
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return {"competitors": [], "financials": []}
    try:
        data = json.loads(match.group(0))
        return {
            "competitors": data.get("competitors", []) if isinstance(data.get("competitors", []), list) else [],
            "financials": data.get("financials", []) if isinstance(data.get("financials", []), list) else [],
        }
    except (json.JSONDecodeError, TypeError):
        return {"competitors": [], "financials": []}

# -------------------- Plotly charts --------------------
def style_figure(fig, title: str, height: int = 500):
    fig.update_layout(title=dict(text=title, x=0.02, xanchor="left"), height=height, **PLOTLY_LAYOUT)
    return fig

def competitor_charts(competitors: list[dict]):
    if not competitors:
        st.info("The sources did not support enough structured competitor data to build a chart. The written analysis and source links are still available.")
        return
    df = pd.DataFrame(competitors)
    required = {"name", "market_presence", "innovation", "offering_breadth"}
    if not required.issubset(df.columns):
        st.info("Competitor comparison data was incomplete, so no chart was rendered.")
        return
    for col in ["market_presence", "innovation", "offering_breadth"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["market_presence", "innovation", "offering_breadth"])
    df = df[
        df["market_presence"].between(1, 10)
        & df["innovation"].between(1, 10)
        & df["offering_breadth"].between(1, 10)
    ]
    if df.empty:
        st.info("No valid competitor ratings were available for plotting.")
        return
    st.caption("The 1–10 axes below are qualitative AI-assisted assessments based on retrieved descriptions—not independently verified market measurements. Hover over points for the rationale and source.")
    if "evidence_note" not in df:
        df["evidence_note"] = ""
    if "source_url" not in df:
        df["source_url"] = ""
    fig3d = px.scatter_3d(
        df, x="market_presence", y="innovation", z="offering_breadth",
        color="name", size_max=16, text="name",
        hover_data={"evidence_note": True, "source_url": True,
                    "market_presence": True, "innovation": True, "offering_breadth": True},
        labels={"market_presence": "Market presence (qualitative)",
                "innovation": "Innovation (qualitative)",
                "offering_breadth": "Offering breadth (qualitative)"},
    )
    fig3d.update_traces(marker=dict(size=7, opacity=0.88), textposition="top center")
    fig3d.update_layout(
        title="Competitor Landscape — Interactive 3D View",
        height=610, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif", color="#18263c"),
        scene=dict(
            xaxis_title="Market presence (1–10)",
            yaxis_title="Innovation (1–10)",
            zaxis_title="Offering breadth (1–10)",
            bgcolor="rgba(245,248,252,0.8)",
        ),
        margin=dict(l=0, r=0, t=65, b=0),
    )
    st.plotly_chart(fig3d, use_container_width=True)
    fig2d = px.scatter(
        df, x="market_presence", y="innovation", color="name",
        size="offering_breadth", hover_name="name",
        hover_data=["evidence_note", "source_url"],
        labels={"market_presence": "Market presence (qualitative)",
                "innovation": "Innovation (qualitative)",
                "offering_breadth": "Offering breadth (bubble size)"},
    )
    style_figure(fig2d, "Competitor Positioning Map", 460)
    st.plotly_chart(fig2d, use_container_width=True)
    with st.expander("View structured competitor ratings"):
        st.dataframe(df, use_container_width=True, hide_index=True)

def financial_charts(financials: list[dict]):
    if not financials:
        st.info("No sufficiently clear financial figures were found in the retrieved sources. We have deliberately not generated placeholder numbers.")
        return
    df = pd.DataFrame(financials)
    required = {"company", "metric", "period", "value", "unit", "source_url"}
    if not required.issubset(df.columns):
        st.info("Financial records were incomplete, so a chart was not rendered.")
        return
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    if df.empty:
        st.info("No valid financial values were found.")
        return
    # Never mix unlike metrics or units in one axis.
    groups = list(df.groupby(["metric", "unit"], dropna=False))
    options = [f"{metric} — {unit}" for (metric, unit), _ in groups]
    selected = st.selectbox("Choose a financial metric and unit", options, key="financial_metric")
    idx = options.index(selected)
    (metric, unit), chart_df = groups[idx]
    fig = px.bar(
        chart_df, x="company", y="value", color="period",
        hover_data=["source_url"],
        labels={"company": "Company", "value": f"{metric} ({unit})", "period": "Period"},
        barmode="group",
    )
    style_figure(fig, f"{metric.title()} Comparison — {unit}", 470)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Only source-reported values are shown. Check the cited source and reporting period before comparing companies.")
    st.dataframe(chart_df, use_container_width=True, hide_index=True)

def trend_visuals(sources: list[dict]):
    # We do not infer numeric time-series from prose. Show source chronology only when dates can be extracted from titles.
    rows = []
    for source in sources:
        text = f"{source.get('title', '')} {source.get('content', '')[:300]}"
        years = re.findall(r"\b(?:20\d{2})\b", text)
        if years:
            year = int(years[0])
            if 2000 <= year <= 2035:
                rows.append({"year": year, "source": source.get("title", "Source"), "url": source.get("url", "")})
    if rows:
        df = pd.DataFrame(rows).drop_duplicates(subset=["year", "url"]).sort_values("year")
        counts = df.groupby("year", as_index=False).size().rename(columns={"size": "source_count"})
        fig = px.bar(
            counts, x="year", y="source_count", text="source_count",
            labels={"year": "Year mentioned in source", "source_count": "Number of retrieved sources"},
        )
        style_figure(fig, "Retrieved Evidence by Mentioned Year", 390)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("This shows the years mentioned in retrieved source text—not market growth, sales, or trend magnitude.")
    else:
        st.info("No reliable dated series was extracted. The trend analysis is available in the written report; no invented time-series chart is shown.")

# -------------------- 3D analytics helpers --------------------
CAMERA_PRESETS = {
    "Default": dict(eye=dict(x=1.6, y=1.6, z=1.1)),
    "Top": dict(eye=dict(x=0.01, y=0.01, z=2.7)),
    "Front": dict(eye=dict(x=0.0, y=-2.5, z=0.35)),
    "Side": dict(eye=dict(x=2.5, y=0.0, z=0.35)),
    "Low angle": dict(eye=dict(x=2.0, y=-1.6, z=0.45)),
}
RATING_COLS = ["market_presence", "innovation", "offering_breadth"]
RATING_LABELS = {"market_presence": "Market presence", "innovation": "Innovation", "offering_breadth": "Offering breadth"}
BAR_FACES_I = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
BAR_FACES_J = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
BAR_FACES_K = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
BAR_VX = [0, 0, 1, 1, 0, 0, 1, 1]
BAR_VY = [0, 1, 1, 0, 0, 1, 1, 0]
BAR_VZ = [0, 0, 0, 0, 1, 1, 1, 1]

def axis_style(title, tickvals=None, ticktext=None, axis_range=None):
    ax = dict(
        title=dict(text=title),
        showbackground=True,
        backgroundcolor="rgba(232,239,248,0.85)",
        gridcolor="#c9d6e8",
        zerolinecolor="#9fb3cc",
        color="#18263c",
    )
    if tickvals is not None:
        ax.update(tickmode="array", tickvals=list(tickvals), ticktext=list(ticktext))
    if axis_range is not None:
        ax["range"] = axis_range
    return ax

def finish_3d(fig, title: str, height: int = 620, scenes=("scene",)):
    """Shared layout + camera-preset buttons (Default / Top / Front / Side / Low angle)."""
    buttons = [
        dict(label=name, method="relayout", args=[{f"{s}.camera": cam for s in scenes}])
        for name, cam in CAMERA_PRESETS.items()
    ]
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left"),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif", color="#18263c"),
        margin=dict(l=0, r=0, t=100, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="#10213a", font_color="white"),
        updatemenus=[dict(
            type="buttons", direction="right", x=0.02, y=1.0, xanchor="left", yanchor="bottom",
            showactive=False, bgcolor="#ffffff", bordercolor="#cfdbea",
            font=dict(color="#18263c", size=11), buttons=buttons,
        )],
    )
    return fig

def wrap_note(text: str, width: int = 55, max_chars: int = 220) -> str:
    text = (text or "").strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    return "<br>".join(textwrap.wrap(text, width)) if text else ""

def clean_competitor_df(competitors):
    """Validate the structured competitor ratings; returns a DataFrame or None."""
    if not competitors:
        return None
    df = pd.DataFrame(competitors)
    if not ({"name"} | set(RATING_COLS)).issubset(df.columns):
        return None
    for col in RATING_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=RATING_COLS)
    for col in RATING_COLS:
        df = df[df[col].between(1, 10)]
    if df.empty:
        return None
    for col in ["evidence_note", "source_url"]:
        if col not in df:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    df["name"] = df["name"].astype(str)
    df = df.drop_duplicates(subset="name").reset_index(drop=True)
    df["composite"] = df[RATING_COLS].mean(axis=1).round(2)
    return df

def competitor_hover(df):
    return [
        f"<b>{r['name']}</b><br>Market presence: {r['market_presence']:.0f}<br>"
        f"Innovation: {r['innovation']:.0f}<br>Offering breadth: {r['offering_breadth']:.0f}<br>"
        f"Composite: {r['composite']:.2f}<br>{wrap_note(r['evidence_note'])}"
        for _, r in df.iterrows()
    ]

def fig_competitor_bubble3d(df):
    """3D bubble scatter with drop-lines to the floor; bubble size and colour = composite rating."""
    fig = go.Figure()
    xs, ys, zs = [], [], []
    for _, r in df.iterrows():
        xs += [r["market_presence"], r["market_presence"], None]
        ys += [r["innovation"], r["innovation"], None]
        zs += [0, r["offering_breadth"], None]
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color="rgba(80,100,130,0.45)", width=3),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=df["market_presence"].tolist(), y=df["innovation"].tolist(), z=df["offering_breadth"].tolist(),
        mode="markers+text", text=df["name"].tolist(), textposition="top center",
        marker=dict(
            size=[float(8 + c * 2.2) for c in df["composite"]],
            color=df["composite"].tolist(), colorscale="Viridis", cmin=1, cmax=10, opacity=0.92,
            line=dict(color="white", width=1),
            colorbar=dict(title=dict(text="Composite<br>(1–10)"), thickness=14, len=0.6),
        ),
        hovertext=competitor_hover(df), hoverinfo="text", showlegend=False,
    ))
    fig.update_layout(scene=dict(
        xaxis=axis_style("Market presence", axis_range=[0, 10.5]),
        yaxis=axis_style("Innovation", axis_range=[0, 10.5]),
        zaxis=axis_style("Offering breadth", axis_range=[0, 10.5]),
        camera=CAMERA_PRESETS["Default"],
    ))
    return finish_3d(fig, "3D Bubble Map — Size & Colour = Composite Rating", 640)

def fig_rating_surface(df):
    """3D surface: competitors (sorted by composite) x rating dimensions -> rating."""
    d = df.sort_values("composite", ascending=False).reset_index(drop=True)
    z = d[RATING_COLS].to_numpy(dtype=float)
    hover = [
        [f"<b>{d.loc[i, 'name']}</b><br>{RATING_LABELS[c]}: {d.loc[i, c]:.0f}" for c in RATING_COLS]
        for i in range(len(d))
    ]
    fig = go.Figure(go.Surface(
        x=[0, 1, 2], y=list(range(len(d))), z=z,
        colorscale="Viridis", cmin=1, cmax=10,
        hovertext=hover, hoverinfo="text",
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
        colorbar=dict(title=dict(text="Rating"), thickness=14, len=0.6),
    ))
    fig.update_layout(scene=dict(
        xaxis=axis_style("Dimension", [0, 1, 2], [RATING_LABELS[c] for c in RATING_COLS]),
        yaxis=axis_style("Competitor (ranked)", list(range(len(d))), d["name"].tolist()),
        zaxis=axis_style("Rating (1–10)", axis_range=[0, 10.5]),
        camera=CAMERA_PRESETS["Low angle"],
    ))
    return finish_3d(fig, "3D Rating Surface — Competitors × Dimensions", 640)

def fig_competitor_subplots(df):
    """Three linked 3D panels: the same competitors coloured by each dimension."""
    scales = ["Viridis", "Plasma", "Cividis"]
    fig = make_subplots(
        rows=1, cols=3, specs=[[{"type": "scene"}] * 3],
        subplot_titles=[f"Coloured by {RATING_LABELS[c]}" for c in RATING_COLS],
        horizontal_spacing=0.01,
    )
    hover = competitor_hover(df)
    scene_names = []
    for n, col in enumerate(RATING_COLS, 1):
        fig.add_trace(go.Scatter3d(
            x=df["market_presence"].tolist(), y=df["innovation"].tolist(), z=df["offering_breadth"].tolist(),
            mode="markers",
            marker=dict(size=7, color=df[col].tolist(), colorscale=scales[n - 1], cmin=1, cmax=10,
                        opacity=0.9, line=dict(color="white", width=0.5)),
            hovertext=hover, hoverinfo="text", showlegend=False,
        ), row=1, col=n)
        key = "scene" if n == 1 else f"scene{n}"
        scene_names.append(key)
        fig.update_layout({key: dict(
            xaxis=axis_style("Presence", axis_range=[0, 10.5]),
            yaxis=axis_style("Innovation", axis_range=[0, 10.5]),
            zaxis=axis_style("Breadth", axis_range=[0, 10.5]),
            camera=CAMERA_PRESETS["Default"],
        )})
    return finish_3d(fig, "3D Subplots — Same Competitors, Three Colour Lenses", 520, tuple(scene_names))

def bars_to_mesh(bars, width: float = 0.7):
    """Turn a list of {x, y, h, c, hover} bars into one Mesh3d's worth of cube geometry."""
    X, Y, Z, I, J, K, T, C = [], [], [], [], [], [], [], []
    for b in bars:
        base = len(X)
        x0, y0 = b["x"] - width / 2, b["y"] - width / 2
        z0, z1 = min(0.0, b["h"]), max(0.0, b["h"])
        if z1 == z0:
            z1 = z0 + 1e-9
        for a in range(8):
            X.append(x0 + BAR_VX[a] * width)
            Y.append(y0 + BAR_VY[a] * width)
            Z.append(z0 if BAR_VZ[a] == 0 else z1)
            C.append(b.get("c", 0))
            T.append(b["hover"])
        I += [base + v for v in BAR_FACES_I]
        J += [base + v for v in BAR_FACES_J]
        K += [base + v for v in BAR_FACES_K]
    return dict(x=X, y=Y, z=Z, i=I, j=J, k=K, intensity=C, text=T)

def fig_financial_bars3d(chart_df, metric: str, unit: str):
    """3D bar chart: company × period -> source-reported value (one metric & unit only)."""
    d = chart_df.copy()
    d["company"] = d["company"].astype(str)
    d["period"] = d["period"].astype(str)
    d = d.drop_duplicates(subset=["company", "period"])
    companies = list(dict.fromkeys(d["company"]))
    periods = list(dict.fromkeys(d["period"]))
    palette = px.colors.qualitative.Bold
    fig = go.Figure()
    for pi, period in enumerate(periods):
        bars = []
        for _, r in d[d["period"] == period].iterrows():
            bars.append(dict(
                x=companies.index(r["company"]), y=pi, h=float(r["value"]),
                hover=f"<b>{r['company']}</b><br>{metric} · {period}<br>{r['value']:,.2f} {unit}<br>Source: {r['source_url']}",
            ))
        m = bars_to_mesh(bars)
        fig.add_trace(go.Mesh3d(
            x=m["x"], y=m["y"], z=m["z"], i=m["i"], j=m["j"], k=m["k"],
            color=palette[pi % len(palette)], opacity=0.92, flatshading=True,
            name=period, showlegend=True, hovertext=m["text"], hoverinfo="text",
        ))
    fig.update_layout(scene=dict(
        xaxis=axis_style("Company", list(range(len(companies))), companies),
        yaxis=axis_style("Period", list(range(len(periods))), periods),
        zaxis=axis_style(f"{metric} ({unit})"),
        camera=CAMERA_PRESETS["Low angle"],
    ))
    return finish_3d(fig, f"3D Bars — {metric.title()} by Company & Period", 640)

def evidence_frame(sources):
    """Metadata of the retrieved sources (query, year mentioned, relevance score, length)."""
    queries = list(dict.fromkeys(s.get("query", "") for s in sources))
    qidx = {q: i for i, q in enumerate(queries)}
    rows = []
    for s in sources:
        content = s.get("content", "") or ""
        text = f"{s.get('title', '')} {content[:300]}"
        years = [int(y) for y in re.findall(r"\b(20\d{2})\b", text)]
        year = years[0] if years and 2000 <= years[0] <= 2035 else None
        score = pd.to_numeric(s.get("score"), errors="coerce")
        rows.append(dict(
            title=s.get("title", "Untitled"), url=s.get("url", ""), query=s.get("query", ""),
            q=qidx.get(s.get("query", ""), 0), year=year, score=score, words=len(content.split()),
        ))
    return pd.DataFrame(rows), queries

def fig_evidence_bars3d(edf, queries):
    d = edf.dropna(subset=["year"]).copy()
    d["year"] = d["year"].astype(int)
    counts = d.groupby(["year", "q"]).size().reset_index(name="count")
    years = sorted(counts["year"].unique())
    yidx = {y: i for i, y in enumerate(years)}
    bars = [dict(
        x=yidx[int(r["year"])], y=int(r["q"]), h=float(r["count"]), c=float(r["count"]),
        hover=f"Year mentioned: {int(r['year'])}<br>Query Q{int(r['q']) + 1}: {wrap_note(queries[int(r['q'])], 45, 120)}<br>Sources: {int(r['count'])}",
    ) for _, r in counts.iterrows()]
    m = bars_to_mesh(bars, 0.65)
    fig = go.Figure(go.Mesh3d(
        x=m["x"], y=m["y"], z=m["z"], i=m["i"], j=m["j"], k=m["k"],
        intensity=m["intensity"], colorscale="Viridis", flatshading=True,
        hovertext=m["text"], hoverinfo="text",
        colorbar=dict(title=dict(text="Sources"), thickness=14, len=0.6),
    ))
    fig.update_layout(scene=dict(
        xaxis=axis_style("Year mentioned in source", list(range(len(years))), [str(y) for y in years]),
        yaxis=axis_style("Research query", list(range(len(queries))), [f"Q{i + 1}" for i in range(len(queries))]),
        zaxis=axis_style("Number of sources"),
        camera=CAMERA_PRESETS["Low angle"],
    ))
    return finish_3d(fig, "3D Evidence Bars — Research Query × Year Mentioned", 620), years

def fig_evidence_surface(edf, queries):
    d = edf.dropna(subset=["year"]).copy()
    d["year"] = d["year"].astype(int)
    years = sorted(d["year"].unique())
    if len(years) < 2 or len(queries) < 2:
        return None
    grid = d.groupby(["q", "year"]).size().unstack(fill_value=0).reindex(
        index=range(len(queries)), columns=years, fill_value=0)
    fig = go.Figure(go.Surface(
        x=list(range(len(years))), y=list(range(len(queries))), z=grid.to_numpy(dtype=float),
        colorscale="Viridis",
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
        hovertemplate="Year idx %{x}<br>Query idx %{y}<br>Sources: %{z}<extra></extra>",
        colorbar=dict(title=dict(text="Sources"), thickness=14, len=0.6),
    ))
    fig.update_layout(scene=dict(
        xaxis=axis_style("Year mentioned", list(range(len(years))), [str(y) for y in years]),
        yaxis=axis_style("Research query", list(range(len(queries))), [f"Q{i + 1}" for i in range(len(queries))]),
        zaxis=axis_style("Number of sources"),
        camera=CAMERA_PRESETS["Default"],
    ))
    return finish_3d(fig, "3D Evidence Density Surface", 620)

def fig_source_scatter3d(edf, queries):
    d = edf.dropna(subset=["score"])
    if d.empty:
        return None
    palette = px.colors.qualitative.Bold
    fig = go.Figure()
    for qi in range(len(queries)):
        sub = d[d["q"] == qi]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter3d(
            x=sub["score"].tolist(), y=sub["q"].tolist(), z=sub["words"].tolist(),
            mode="markers", name=f"Q{qi + 1}",
            marker=dict(size=6, color=palette[qi % len(palette)], opacity=0.88, line=dict(color="white", width=0.5)),
            hovertext=[f"<b>{wrap_note(t, 50, 110)}</b><br>Relevance: {s:.3f}<br>Words retrieved: {w}<br>{u}"
                       for t, s, w, u in zip(sub["title"], sub["score"], sub["words"], sub["url"])],
            hoverinfo="text",
        ))
    fig.update_layout(scene=dict(
        xaxis=axis_style("Search relevance score"),
        yaxis=axis_style("Research query", list(range(len(queries))), [f"Q{i + 1}" for i in range(len(queries))]),
        zaxis=axis_style("Words retrieved"),
        camera=CAMERA_PRESETS["Default"],
    ))
    return finish_3d(fig, "3D Source Map — Relevance × Query × Content Length", 620)


# -------------------- 3D Analytics Lab (Streamlit layout) --------------------
def three_d_lab(chart_data: dict, sources: list[dict]):
    st.markdown("### 🧊 3D Analytics Lab")
    st.caption(
        "Every chart here is built only from data collected in this run: structured competitor ratings, "
        "source-reported financial figures, and metadata of the retrieved sources. Use the camera buttons "
        "above each chart (Default / Top / Front / Side / Low angle), or drag to rotate, scroll to zoom."
    )
    sub_comp, sub_fin, sub_evi = st.tabs(["🏢 Competitor 3D", "💰 Financial 3D", "🔎 Evidence 3D"])

    # ---- Competitor 3D ----
    with sub_comp:
        df = clean_competitor_df(chart_data.get("competitors", []))
        if df is None:
            st.info("The sources did not support enough structured competitor ratings for 3D competitor charts. The Evidence 3D tab is still available.")
        else:
            st.caption("Ratings (1–10) are qualitative, AI-assisted assessments inferred from retrieved descriptions, not verified market measurements. Hover to see the rationale.")
            st.plotly_chart(fig_competitor_bubble3d(df), use_container_width=True, key="lab_bubble3d")
            if len(df) >= 2:
                st.plotly_chart(fig_rating_surface(df), use_container_width=True, key="lab_rating_surface")
                st.caption("Only the grid points are real ratings; the surface just joins them so patterns across competitors are easier to see. Competitors are ranked by composite rating (average of the three dimensions).")
            else:
                st.info("At least two competitors are needed for the 3D rating surface.")
            st.plotly_chart(fig_competitor_subplots(df), use_container_width=True, key="lab_comp_subplots")
            with st.expander("Competitor ratings used in these charts"):
                st.dataframe(df[["name"] + RATING_COLS + ["composite", "evidence_note", "source_url"]],
                             use_container_width=True, hide_index=True)

    # ---- Financial 3D ----
    with sub_fin:
        financials = chart_data.get("financials", [])
        fdf = pd.DataFrame(financials) if financials else pd.DataFrame()
        required = {"company", "metric", "period", "value", "unit", "source_url"}
        if fdf.empty or not required.issubset(fdf.columns):
            st.info("No sufficiently clear financial figures were found in the retrieved sources, so no 3D financial chart is shown. Placeholder numbers are deliberately not generated.")
        else:
            fdf["value"] = pd.to_numeric(fdf["value"], errors="coerce")
            fdf = fdf.dropna(subset=["value"])
            if fdf.empty:
                st.info("No valid financial values were found.")
            else:
                groups = list(fdf.groupby(["metric", "unit"], dropna=False))
                options = [f"{metric} — {unit}" for (metric, unit), _ in groups]
                selected = st.selectbox("Choose a financial metric and unit", options, key="lab_fin_metric")
                (metric, unit), chart_df = groups[options.index(selected)]
                st.plotly_chart(fig_financial_bars3d(chart_df, str(metric), str(unit)),
                                use_container_width=True, key="lab_fin_bars3d")
                st.caption("Only source-reported values are plotted, one metric and unit at a time. Check the cited source and reporting period before comparing companies.")

    # ---- Evidence 3D ----
    with sub_evi:
        if not sources:
            st.info("No sources were returned, so no evidence charts can be drawn.")
        else:
            edf, queries = evidence_frame(sources)
            st.caption("These charts describe the retrieved evidence itself (which research query found what, when it is dated, how relevant and how long it is). They are not market-growth or sales figures.")
            if edf["year"].notna().any():
                bars_fig, _ = fig_evidence_bars3d(edf, queries)
                st.plotly_chart(bars_fig, use_container_width=True, key="lab_evi_bars3d")
                surf = fig_evidence_surface(edf, queries)
                if surf is not None:
                    st.plotly_chart(surf, use_container_width=True, key="lab_evi_surface")
            else:
                st.info("No source mentioned a clear year, so the year-based 3D bar and surface charts are skipped.")
            scat = fig_source_scatter3d(edf, queries)
            if scat is not None:
                st.plotly_chart(scat, use_container_width=True, key="lab_src_scatter3d")
            with st.expander("Research queries (Q1, Q2, …)"):
                for i, q in enumerate(queries, 1):
                    st.markdown(f"**Q{i}.** {q}")

# -------------------- User interface --------------------
with st.sidebar:
    st.markdown("## 📊 Market Intelligence Studio")
    st.markdown("**Research • Compare • Discover**")
    st.markdown("---")
    st.markdown("### Research workflow")
    st.markdown("1. Plan targeted queries\n2. Retrieve web evidence\n3. Analyze competitors\n4. Identify trends\n5. Review financial evidence\n6. Synthesize report")
    st.markdown("---")
    st.caption("Powered by LangGraph · OpenAI · Tavily · Plotly")

st.markdown("""
<div class="hero">
  <h1>Market Intelligence Studio</h1>
  <p>Turn web evidence into a structured market brief—with interactive, source-aware visual analytics.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.5, 1])
with left:
    st.markdown("### Start a research brief")
    st.write("Enter a company, product, or industry. The studio researches the topic and builds a report with competitor and financial visualizations when the evidence supports them.")
with right:
    st.markdown("#### Evidence-first analysis")
    st.markdown("**No fabricated metrics.** Qualitative ratings are labeled, and numerical charts use source-reported values.")

with st.form("research_form"):
    company = st.text_input("Company / product / industry", placeholder="e.g., NVIDIA, electric vehicles, Indian quick commerce")
    topic = st.text_area(
        "What do you want to understand?",
        value="Market position, competitors, emerging trends, opportunities, risks, and financial snapshot",
        height=90,
    )
    submitted = st.form_submit_button("🚀 Run market intelligence", use_container_width=True, type="primary")

if submitted:
    if not OPENAI_API_KEY or not TAVILY_API_KEY:
        st.error("API keys are missing. Add OPENAI_API_KEY and TAVILY_API_KEY to your .env file, then restart Streamlit.")
    elif not company.strip() or not topic.strip():
        st.warning("Please enter both a company/topic and a research focus.")
    else:
        try:
            graph = build_graph()
            initial_state = {"company": company.strip(), "topic": topic.strip(), "sources": []}
            with st.spinner("Researching the web, analyzing evidence, and preparing your report…"):
                result = graph.invoke(initial_state)
                chart_data = extract_chart_data(company.strip(), topic.strip(), result.get("sources", []))
                result["chart_data"] = chart_data
                result["research_company"] = company.strip()
                result["research_topic"] = topic.strip()
                st.session_state["research_result"] = result
            st.success(f"Research complete: {len(result.get('sources', []))} unique sources collected.")
        except Exception as exc:
            st.error(f"Research could not be completed: {exc}")
            st.caption("Check that your API keys are valid, the required packages are installed, and your API account has available usage.")

result = st.session_state.get("research_result")
if result:
    sources = result.get("sources", [])
    chart_data = result.get("chart_data", {})
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Sources collected", len(sources))
    m2.metric("Competitors identified", len(chart_data.get("competitors", [])))
    m3.metric("Financial records", len(chart_data.get("financials", [])))

    tabs = st.tabs(["📌 Executive Report", "🏢 Competitors", "📈 Trends", "💰 Financials", "🧊 3D Lab", "🔎 Sources"])
    with tabs[0]:
        st.markdown(result.get("report", "No report was generated."))
        st.download_button(
            "⬇️ Download report as Markdown",
            data=result.get("report", ""),
            file_name=f"{re.sub(r'[^a-zA-Z0-9_-]+', '_', result.get('research_company', 'market_report'))}_report.md",
            mime="text/markdown",
        )
    with tabs[1]:
        st.markdown("### Competitive landscape")
        st.markdown(result.get("competitor_analysis", "No competitor analysis available."))
        competitor_charts(chart_data.get("competitors", []))
    with tabs[2]:
        st.markdown("### Market trends and opportunities")
        st.markdown(result.get("trends_analysis", "No trends analysis available."))
        trend_visuals(sources)
    with tabs[3]:
        st.markdown("### Financial snapshot")
        st.markdown(result.get("financial_analysis", "No financial analysis available."))
        financial_charts(chart_data.get("financials", []))
    with tabs[4]:
        three_d_lab(chart_data, sources)
    with tabs[5]:
        st.markdown("### Retrieved evidence")
        if sources:
            source_df = pd.DataFrame(sources)
            visible_cols = [c for c in ["title", "query", "url", "score"] if c in source_df.columns]
            st.dataframe(
                source_df[visible_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "url": st.column_config.LinkColumn("Source URL", display_text="Open source"),
                    "score": st.column_config.NumberColumn("Search relevance", format="%.3f"),
                },
            )
            st.download_button(
                "⬇️ Download source list (CSV)",
                data=source_df.to_csv(index=False).encode("utf-8"),
                file_name="market_intelligence_sources.csv",
                mime="text/csv",
            )
        else:
            st.info("No sources were returned.")
else:
    st.markdown("---")
    st.markdown("### What the visual analytics can show")
    a, b, c = st.columns(3)
    with a:
        st.markdown("#### 🧭 3D competitor map")
        st.write("Explore competitors as 3D bubbles, rating surfaces, financial 3D bars and evidence maps in the 3D Lab tab. Rotate, zoom, and inspect source notes.")
    with b:
        st.markdown("#### 📊 Financial comparisons")
        st.write("Compare values only when retrieved evidence includes figures, units, and periods.")
    with c:
        st.markdown("#### 🗓️ Evidence chronology")
        st.write("Inspect the years mentioned in sources without misrepresenting them as a market-growth time series.")
    st.caption("Run a research brief to populate the report and charts. Some visualizations may remain unavailable when the retrieved evidence does not support them.")