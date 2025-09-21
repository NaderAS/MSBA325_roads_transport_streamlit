# PKGCube Roads & Transport – Streamlit app (polished)
# ----------------------------------------------------
# Interactions:
# 1) Filter by Governorate (and optional Towns)
# 2) Show % vs. raw counts
# 3) Pick which road types to include (Main/Secondary/Agricultural)
# 4) Optional filter: keep towns that have dedicated bus stops
# + Theme switch (Light/Dark) and aesthetic upgrades

import re, csv
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

CSV_URL = "https://linked.aub.edu.lb/pkgcube/data/b97344d097e356329ba1e182721382e7.csv"

# ---------- visual style ----------
APP_TITLE = "PKGCube Insights: Roads & Transport in Lebanon"
PRIMARY_FONT = "Inter"

# Consistent semantic colors
STATE_COLORS = {
    "good": "#2ecc71",        # green
    "acceptable": "#f1c40f",  # amber
    "bad": "#e74c3c"          # red
}

PLOTLY_TEMPLATES = {
    "Light": "simple_white",
    "Dark": "plotly_dark",
}

# ------------------------- Data loading & prep -------------------------

@st.cache_data(show_spinner=False)
def load_csv(url: str) -> pd.DataFrame:
    for sep in [",",";","|\t","\t","|"]:
        for enc in ["utf-8","utf-8-sig","latin-1"]:
            try:
                df = pd.read_csv(
                    url, sep=sep, engine="python", encoding=enc,
                    on_bad_lines="skip", quoting=csv.QUOTE_MINIMAL, escapechar="\\"
                )
                if df.shape[1] > 1:
                    return df
            except Exception:
                pass
    for enc in ["utf-8","utf-8-sig","latin-1"]:
        try:
            df = pd.read_csv(
                url, sep=None, engine="python", encoding=enc,
                on_bad_lines="skip", quoting=csv.QUOTE_MINIMAL, escapechar="\\"
            )
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    # last resort
    raw = pd.read_csv(url, header=None, engine="python", encoding="latin-1",
                      on_bad_lines="skip", quoting=csv.QUOTE_MINIMAL, escapechar="\\")
    return raw[0].str.split(",", expand=True)

@st.cache_data(show_spinner=False)
def prep() -> pd.DataFrame:
    df = load_csv(CSV_URL).copy()
    df.columns = [re.sub(r"\s+"," ", str(c)).strip() for c in df.columns]
    town_col = next((c for c in df.columns if c.lower()=="town"), None)
    ref_col  = next((c for c in df.columns if "refarea" in c.lower()), None)

    # binary indicator columns
    skip_like = {"town","refarea","observation uri","publisher","dataset","references"}
    indicator_cols = []
    for c in df.columns:
        if c.lower() in skip_like:
            continue
        ser = pd.to_numeric(df[c], errors="coerce")
        vals = ser.dropna().unique()
        if len(vals) and set(vals).issubset({0,1}):
            indicator_cols.append(c)
    if indicator_cols:
        df[indicator_cols] = df[indicator_cols].apply(pd.to_numeric, errors="coerce").fillna(0).clip(0,1)

    # governorate (clean from DBpedia URL)
    def clean_gov(x):
        if pd.isna(x): return np.nan
        s = str(x).split("/")[-1].replace("_"," ")
        s = re.sub(r"[-+]", " ", s).strip()
        s = re.sub(r"\s*(District|Governorate)\s*$", "", s, flags=re.I)
        return s
    df["Governorate"] = df[ref_col].map(clean_gov) if ref_col else np.nan

    # helpers
    def cols_containing(phrase):
        p = phrase.lower()
        return [c for c in indicator_cols if p in c.lower()]

    groups = {
        "main": cols_containing("state of the main roads"),
        "secondary": cols_containing("state of the secondary roads"),
        "agri": cols_containing("state of agricultural roads"),
        "bus_stops": cols_containing("dedicated bus stops"),
        "buses": cols_containing("main means of public transport - buses"),
        "vans": cols_containing("main means of public transport - vans"),
        "taxis": cols_containing("main means of public transport - taxis"),
    }

    return df, town_col, groups

df, TOWN, G = prep()

# ------------------------------ UI -----------------------------------

st.set_page_config(page_title="PKGCube – Roads & Transport", page_icon="🛣️", layout="wide")

# Small CSS polish (KPI cards + spacing)
st.markdown("""
<style>
.kpi {border-radius:16px; padding:16px 18px; border:1px solid rgba(120,120,120,.2);}
.kpi h3 {font-size:16px; margin:0; opacity:.8;}
.kpi p  {font-size:28px; margin:4px 0 0 0; font-weight:700;}
.block  {margin-top: 0.8rem;}
</style>
""", unsafe_allow_html=True)

# Sidebar filters
st.sidebar.header("Filters")

# Theme switcher
st.sidebar.markdown("### Theme")
theme_choice = st.sidebar.radio("", ["Light", "Dark"], index=0, horizontal=True)
plotly_template = PLOTLY_TEMPLATES[theme_choice]

govs = sorted([g for g in df["Governorate"].dropna().unique()])
sel_govs = st.sidebar.multiselect("Governorates", govs, default=govs[:5] if govs else [])
if sel_govs:
    df = df[df["Governorate"].isin(sel_govs)]

# Optional town filter (only show if Town exists and we narrowed govs)
if TOWN and len(df) > 0:
    towns = sorted(df[TOWN].astype(str).unique().tolist())
    default_towns = towns if len(towns) <= 30 else towns[:30]
    sel_towns = st.sidebar.multiselect("Towns (optional)", towns, default=default_towns)
    if sel_towns:
        df = df[df[TOWN].astype(str).isin(sel_towns)]

only_with_bus_stops = st.sidebar.checkbox("Keep towns that have dedicated bus stops", value=False)
if only_with_bus_stops and G["bus_stops"]:
    df = df[df[G["bus_stops"]].sum(axis=1).clip(0,1) == 1]

show_percent = st.sidebar.radio("Normalize values as", ["Percent of towns", "Counts"], index=0)

road_types = st.sidebar.multiselect(
    "Road types to include",
    ["Main roads","Secondary roads","Agricultural roads"],
    default=["Main roads","Secondary roads","Agricultural roads"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Data source: PKGCube CSV")

# Title & KPIs
st.title(APP_TITLE)
st.caption("Interactive view of road quality and public transport coverage from PKGCube data.")

total_towns = int(df.shape[0])
buses_share = (df[G["buses"]].sum(axis=1).clip(0,1).sum() / total_towns * 100) if G["buses"] and total_towns else 0
vans_share  = (df[G["vans"]].sum(axis=1).clip(0,1).sum()  / total_towns * 100) if G["vans"]  and total_towns else 0
taxis_share = (df[G["taxis"]].sum(axis=1).clip(0,1).sum() / total_towns * 100) if G["taxis"] and total_towns else 0

col1, col2, col3, col4 = st.columns(4)
col1.markdown(f"""<div class='kpi'><h3>Towns in view</h3><p>{total_towns:,}</p></div>""", unsafe_allow_html=True)
col2.markdown(f"""<div class='kpi'><h3>Towns with buses</h3><p>{buses_share:.0f}%</p></div>""", unsafe_allow_html=True)
col3.markdown(f"""<div class='kpi'><h3>Towns with vans</h3><p>{vans_share:.0f}%</p></div>""", unsafe_allow_html=True)
col4.markdown(f"""<div class='kpi'><h3>Towns with taxis</h3><p>{taxis_share:.0f}%</p></div>""", unsafe_allow_html=True)

# ------------------------- VISUAL 1: Road quality mix -------------------------

def summarize_road_mix(cols, label):
    if not cols or len(df)==0:
        return None
    totals = pd.Series({c: int(df[c].sum()) for c in cols}).rename("Count").reset_index()
    totals.rename(columns={"index":"Indicator"}, inplace=True)
    totals["State"] = totals["Indicator"].str.extract(r"-\s*(good|acceptable|bad)", expand=False)
    s = totals.groupby("State", dropna=False)["Count"].sum()
    s = s.reindex(["good","acceptable","bad"], fill_value=0).reset_index()
    s["Road Type"] = label
    return s

parts = []
if "Main roads" in road_types:      parts.append(summarize_road_mix(G["main"], "Main roads"))
if "Secondary roads" in road_types: parts.append(summarize_road_mix(G["secondary"], "Secondary roads"))
if "Agricultural roads" in road_types: parts.append(summarize_road_mix(G["agri"], "Agricultural roads"))
parts = [p for p in parts if p is not None and not p.empty]

st.subheader("Road quality mix by road type")
st.caption("Use the sidebar to filter locations and choose road types.")

if parts:
    mix = pd.concat(parts, ignore_index=True)
    if show_percent == "Percent of towns":
        mix["Value"] = mix.groupby("Road Type")["Count"].transform(lambda x: 100 * x / (x.sum() if x.sum() else 1))
        ylab = "Share (%)"
    else:
        mix["Value"] = mix["Count"]
        ylab = "Count"

    fig1 = px.bar(
        mix,
        x="Road Type",
        y="Value",
        color="State",
        barmode="stack",
        category_orders={
            "State": ["good","acceptable","bad"],
            "Road Type": ["Main roads","Secondary roads","Agricultural roads"]
        },
        color_discrete_map=STATE_COLORS,
        labels={"Value": ylab, "State":"Quality"},
        template=plotly_template,
    )
    fig1.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title="",
        font=dict(family=PRIMARY_FONT, size=14),
    )
    fig1.update_yaxes(showgrid=True, gridcolor="rgba(180,180,180,.25)")
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.info("No road indicators available with the current filters.")

# ------------------- VISUAL 2: Public transport availability ------------------

st.subheader("Public transport availability across towns")
st.caption("Compare coverage of buses, vans, and taxis in the filtered towns.")

transport = []
if G["buses"]: transport.append(("Buses",  int(df[G["buses"]].sum(axis=1).clip(0,1).sum())))
if G["vans"]:  transport.append(("Vans",   int(df[G["vans"]].sum(axis=1).clip(0,1).sum())))
if G["taxis"]: transport.append(("Taxis",  int(df[G["taxis"]].sum(axis=1).clip(0,1).sum())))

if transport:
    tr = pd.DataFrame(transport, columns=["Mode","Towns with Mode"])
    if show_percent == "Percent of towns":
        tr["Value"] = 100 * tr["Towns with Mode"] / max(total_towns, 1)
        ylab2 = "Share (%)"
        txt = tr["Value"].round(0).astype(int).astype(str) + "%"
        ymax = max(100, tr["Value"].max()*1.1)
    else:
        tr["Value"] = tr["Towns with Mode"]
        ylab2 = "Towns"
        txt = tr["Value"].round(0).astype(int)
        ymax = max(tr["Value"].max()*1.15, 10)

    fig2 = px.bar(
        tr, x="Mode", y="Value", text=txt,
        labels={"Value": ylab2},
        template=plotly_template,
    )
    fig2.update_traces(textposition="outside", cliponaxis=False)
    fig2.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(family=PRIMARY_FONT, size=14),
        yaxis=dict(range=[0, ymax])
    )
    fig2.update_yaxes(showgrid=True, gridcolor="rgba(180,180,180,.25)")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No transport indicators available with the current filters.")

# ------------------------------- Footer --------------------------------

st.markdown("---")
st.caption(
    "Tips: pick a governorate in the sidebar, switch between % and counts, "
    "and toggle road types to compare mixes. Optional: keep only towns with bus stops."
)
