"""
NEXMAINT - Predictive Maintenance Dashboard (Streamlit, pure Python)

Run:
    pip install -r requirements.txt
    streamlit run app.py

Put your trained files in this same folder before running:
    random_forest_model.pkl
    preprocessor.pkl

Also keep the .streamlit/config.toml file next to app.py — it sets the
dark theme for Streamlit's built-in widgets (buttons, dataframe, inputs)
so they match the custom CSS instead of clashing with it.

If the model files are missing, the app still runs using a simple
fallback risk formula, so you can see the GUI before wiring up the
real model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
from datetime import datetime, timedelta

# =========================================================
# PAGE CONFIG + GLOBAL STYLE
# =========================================================
st.set_page_config(
    page_title="NEXMAINT - Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
    .stApp { background-color: #0b1220; }
    section[data-testid="stSidebar"] { background-color: #0e1626; border-right: 1px solid #1f2c45; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }

    /* space out columns and stacked blocks consistently */
    div[data-testid="stHorizontalBlock"] { gap: 1.1rem; }
    div[data-testid="stVerticalBlockBorderWrapper"] { margin-bottom: 1.1rem; }

    /* bordered containers = our "cards" */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: #111b2e !important;
        border-color: #1f2c45 !important;
        border-radius: 12px !important;
        padding: 4px 6px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] h4 {
        color: #8ea0bf; font-size: 12.5px; text-transform: uppercase;
        letter-spacing: .4px; margin: 4px 0 10px 0;
    }

    div[data-testid="stMetric"] {
        background-color: #111b2e;
        border: 1px solid #1f2c45;
        border-radius: 12px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricLabel"] { color: #8ea0bf; font-size: 12px; text-transform: uppercase; }

    .pill { display:inline-block; padding:3px 12px; border-radius:20px; font-size:12px; font-weight:700; }
    .pill-low { background: rgba(34,197,94,.15); color:#22c55e; }
    .pill-medium { background: rgba(245,158,11,.15); color:#f59e0b; }
    .pill-high { background: rgba(239,68,68,.15); color:#ef4444; }
    .pill-critical { background: rgba(239,68,68,.25); color:#fca5a5; }

    .alert-row { display:flex; gap:10px; padding:9px 0; border-bottom:1px solid #16213a; align-items:flex-start;}
    .alert-dot { width:8px; height:8px; border-radius:50%; margin-top:6px; flex-shrink:0; }

    h1, h2, h3 { color: #e7ecf5; }
    hr { border-color: #1f2c45; }

    /* tighten default vertical gaps between stacked elements inside a card */
    div[data-testid="stVerticalBlock"] { gap: 0.4rem; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
# Neon accent palette against the deep navy/slate background
COLORS = {"Low": "#39ff88", "Medium": "#ffb020", "High": "#ff3b3b", "Critical": "#ff6b6b"}
ACCENT = "#3b82f6"
CHART_MARGIN = dict(l=10, r=10, t=30, b=10)


# =========================================================
# MODEL LOADING
# =========================================================



MODEL_PATH = os.path.join("models", "random_forest_model.pkl")
PREPROCESSOR_PATH = os.path.join("models", "preprocessor.pkl")

NUM_COLS = ["volt", "rotate", "pressure", "vibration", "age", "vibration_roll3", "error_count"]
CAT_COLS = ["model", "errorID", "comp", "age_group"]
ALL_COLS = NUM_COLS + CAT_COLS


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return model, preprocessor, None
    return None, None, "Model files not found — using fallback risk formula instead of your trained model."


model, preprocessor, model_load_msg = load_model()


def age_group(age):
    if age <= 5:
        return "New"
    elif age <= 10:
        return "Medium"
    return "Old"


def risk_level_from_prob(prob_pct):
    if prob_pct < 25:
        return "Low", "Healthy"
    elif prob_pct < 50:
        return "Medium", "Warning"
    elif prob_pct < 75:
        return "High", "Warning"
    return "Critical", "Critical"


def predict_failure(row: dict):
    """Runs the real model if available, else a transparent fallback formula."""
    if model is not None and preprocessor is not None:
        record = {
            "volt": row["volt"], "rotate": row["rotate"], "pressure": row["pressure"],
            "vibration": row["vibration"], "age": row["age"],
            "vibration_roll3": row.get("vibration_roll3", row["vibration"]),
            "error_count": row.get("error_count", 0),
            "model": row["model"], "errorID": row.get("errorID", "no error"),
            "comp": row.get("comp", "no maintenance"), "age_group": age_group(row["age"]),
        }
        df = pd.DataFrame([record])[ALL_COLS]
        X = preprocessor.transform(df)
        prob = float(model.predict_proba(X)[:, 1][0]) * 100
    else:
        prob = min(100, max(0,
            (row["vibration"] - 30) * 1.5 + (row["age"]) * 1.2 + row.get("error_count", 0) * 6))
    risk, status = risk_level_from_prob(prob)
    return round(prob, 1), round(100 - prob, 1), risk, status


# =========================================================
# MOCK FLEET DATA (replace with your real telemetry later)
# =========================================================
@st.cache_data
def generate_fleet(n=30, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(1, n + 1):
        row = {
            "machineID": f"M-{100+i}",
            "model": rng.choice(["model1", "model2", "model3", "model4"]),
            "age": int(rng.integers(1, 20)),
            "volt": round(rng.normal(170, 12), 1),
            "rotate": round(rng.normal(450, 30), 1),
            "pressure": round(rng.normal(100, 12), 1),
            "vibration": round(rng.normal(42, 12), 1),
            "error_count": int(rng.integers(0, 5)),
            "errorID": rng.choice(["no error", "error1", "error2", "error3"], p=[.6, .15, .15, .1]),
            "comp": "no maintenance",
            "location": rng.choice(["Plant A / Line 1", "Plant A / Line 3", "Plant B / Line 2"]),
            "installed": (datetime(2020, 1, 1) + timedelta(days=int(rng.integers(0, 1800)))).strftime("%b %d, %Y"),
            "last_maint": (datetime(2025, 3, 1) + timedelta(days=int(rng.integers(0, 60)))).strftime("%b %d, %Y"),
        }
        fail_prob, health, risk, status = predict_failure(row)
        row.update(fail_prob=fail_prob, health=health, risk=risk, status=status)
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data
def generate_alerts(fleet_df):
    msgs = ["High vibration detected", "High temperature detected", "Abnormal pressure detected", "Error count exceeded"]
    rows = []
    pool = fleet_df.sort_values("fail_prob", ascending=False).head(8)
    for i, (_, m) in enumerate(pool.iterrows()):
        rows.append({
            "machineID": m["machineID"],
            "message": np.random.choice(msgs),
            "risk": m["risk"],
            "time": (datetime.now() - timedelta(minutes=15 * i)).strftime("%b %d, %I:%M %p"),
            "status": np.random.choice(["New", "In Progress", "Resolved"], p=[.6, .25, .15]),
        })
    return pd.DataFrame(rows)


fleet = generate_fleet()
alerts = generate_alerts(fleet)

if "selected_machine" not in st.session_state:
    st.session_state.selected_machine = fleet.iloc[0]["machineID"]


# =========================================================
# REUSABLE UI HELPERS
# =========================================================
def pill(text, risk):
    cls = f"pill-{risk.lower()}"
    return f'<span class="pill {cls}">{text}</span>'


def gauge(value, title, color, suffix=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix, "font": {"size": 28, "color": color}},
        title={"text": title, "font": {"size": 13, "color": "#8ea0bf"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8ea0bf", "tickfont": {"size": 9}},
            "bar": {"color": color},
            "bgcolor": "#1c2740",
            "borderwidth": 0,
        },
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       height=200, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def styled_chart_layout(fig, height=260):
    fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       height=height, margin=CHART_MARGIN, font_color="#e7ecf5")
    return fig


def risk_table(df):
    show = df[["machineID", "model", "age", "health", "fail_prob", "risk", "status"]].copy()
    show.columns = ["Machine ID", "Model", "Age", "Health", "Fail Prob %", "Risk", "Status"]
    return show


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
NAV_ITEMS = [
    ("🏠", "Dashboard"),
    ("⚙️", "Machines"),
    ("🔎", "Machine Details"),
    ("📈", "Analytics"),
    ("🚨", "Alerts"),
    ("📄", "Reports"),
    ("🛠️", "Settings / About"),
]
NAV_LABELS = [f"{icon}  {label}" for icon, label in NAV_ITEMS]
LABEL_TO_PAGE = dict(zip(NAV_LABELS, [label for _, label in NAV_ITEMS]))

with st.sidebar:
    st.markdown("### ⚙️ NEXMAINT")
    st.caption("Predictive Maintenance")
    st.markdown("---")
    chosen_label = st.radio("Navigate", NAV_LABELS, label_visibility="collapsed")
    page = LABEL_TO_PAGE[chosen_label]
    st.markdown("---")
    if model_load_msg:
        st.warning(model_load_msg, icon="⚠️")
    else:
        st.success("Model loaded ✓", icon="✅")
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("⎋ Logout", use_container_width=True)


# =========================================================
# PAGE: DASHBOARD
# =========================================================
if page == "Dashboard":
    st.title("Dashboard")
    healthy = (fleet["status"] == "Healthy").sum()
    medium = (fleet["risk"] == "Medium").sum()
    high = fleet["risk"].isin(["High", "Critical"]).sum()

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    c1.metric("Total Machines", len(fleet))
    c2.metric("Healthy Machines", healthy, f"{healthy/len(fleet)*100:.0f}%")
    c3.metric("Medium Risk", medium, f"{medium/len(fleet)*100:.0f}%")
    c4.metric("High / Critical Risk", high, f"{high/len(fleet)*100:.0f}%", delta_color="inverse")

    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        with st.container(border=True):
            st.markdown("#### Overall Health Score")
            overall_health = round(fleet["health"].mean(), 1)
            st.plotly_chart(gauge(overall_health, "Fleet Health", "#3b82f6"), use_container_width=True)
    with col2:
        with st.container(border=True):
            st.markdown("#### Risk Level Distribution")
            counts = fleet["risk"].value_counts().reindex(["Low", "Medium", "High", "Critical"]).fillna(0)
            fig = px.pie(values=counts.values, names=counts.index, hole=0.6,
                         color=counts.index, color_discrete_map=COLORS)
            st.plotly_chart(styled_chart_layout(fig, 220), use_container_width=True)
    with col3:
        with st.container(border=True):
            st.markdown("#### Failure Probability Trend")
            trend = pd.DataFrame({"day": pd.date_range(end=datetime.now(), periods=7),
                                   "avg_fail_prob": np.clip(fleet["fail_prob"].mean() + np.random.normal(0, 5, 7).cumsum(), 0, 100)})
            fig = px.line(trend, x="day", y="avg_fail_prob")
            fig.update_traces(line_color="#3b82f6")
            st.plotly_chart(styled_chart_layout(fig, 220), use_container_width=True)

    col1, col2 = st.columns([2, 1], gap="medium")
    with col1:
        with st.container(border=True):
            st.markdown("#### Recent Predictions")
            st.dataframe(risk_table(fleet.head(6)), use_container_width=True, hide_index=True)
    with col2:
        with st.container(border=True):
            st.markdown("#### Top Alerts")
            for _, a in alerts.head(4).iterrows():
                st.markdown(
                    f'<div class="alert-row"><div class="alert-dot" style="background:{COLORS[a["risk"]]}"></div>'
                    f'<div style="flex:1"><div>{a["message"]} in <b>{a["machineID"]}</b></div>'
                    f'<div style="color:#8ea0bf;font-size:11px">{a["time"]}</div></div>'
                    f'{pill(a["risk"], a["risk"])}</div>', unsafe_allow_html=True)


# =========================================================
# PAGE: MACHINES
# =========================================================
elif page == "Machines":
    st.title("Machines")
    with st.container(border=True):
        top1, top2 = st.columns([3, 1])
        search = top1.text_input("🔍 Search Machine ID")
        risk_filter = top2.selectbox("Filter risk", ["All"] + list(COLORS.keys()))

        view = fleet.copy()
        if search:
            view = view[view["machineID"].str.contains(search, case=False)]
        if risk_filter != "All":
            view = view[view["risk"] == risk_filter]

        PAGE_SIZE = 8
        total_pages = max(1, -(-len(view) // PAGE_SIZE))  # ceil division
        page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
        start, end = (page_num - 1) * PAGE_SIZE, (page_num - 1) * PAGE_SIZE + PAGE_SIZE
        page_view = view.iloc[start:end]

        st.dataframe(risk_table(page_view), use_container_width=True, hide_index=True)
        st.caption(f"Showing {start+1}–{min(end, len(view))} of {len(view)} machines · Page {page_num} of {total_pages}")

        pick = st.selectbox("Open machine details for:", page_view["machineID"] if len(page_view) else view["machineID"])
        if st.button("View Selected Machine →"):
            st.session_state.selected_machine = pick
            st.info("Go to the **Machine Details** page from the sidebar to see it.")


# =========================================================
# PAGE: MACHINE DETAILS
# =========================================================
elif page == "Machine Details":
    st.title("Machine Details")
    machine_id = st.selectbox("Machine ID", fleet["machineID"],
                               index=list(fleet["machineID"]).index(st.session_state.selected_machine))
    st.session_state.selected_machine = machine_id
    m = fleet[fleet["machineID"] == machine_id].iloc[0]

    st.markdown(f"### {m['machineID']}  {pill(m['risk'].upper()+' RISK', m['risk'])}", unsafe_allow_html=True)

    tab_overview, tab_sensors, tab_ai, tab_maint, tab_history = st.tabs(
        ["Overview", "Sensors", "AI Prediction", "Maintenance", "History"]
    )

    with tab_overview:
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            with st.container(border=True):
                st.plotly_chart(gauge(m["health"], "Health Score", ACCENT), use_container_width=True)
        with c2:
            with st.container(border=True):
                st.plotly_chart(gauge(m["fail_prob"], "Failure Probability", COLORS[m["risk"]], suffix="%"), use_container_width=True)
        with c3:
            with st.container(border=True):
                st.markdown("#### Machine Information")
                st.write(f"**Model:** {m['model']}")
                st.write(f"**Age:** {m['age']} years")
                st.write(f"**Installed:** {m['installed']}")
                st.write(f"**Location:** {m['location']}")
                st.write(f"**Last Maintenance:** {m['last_maint']}")

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            with st.container(border=True):
                st.markdown("#### Top Contributing Factors")
                factors = pd.DataFrame({
                    "factor": ["Vibration", "Pressure", "Error Count", "Machine Age", "Rotation"],
                    "impact": [m["vibration"], m["pressure"] / 2, m["error_count"] * 15, m["age"] * 3, m["rotate"] / 10],
                }).sort_values("impact", ascending=True)
                fig = px.bar(factors, x="impact", y="factor", orientation="h")
                fig.update_traces(marker_color=COLORS["High"])
                fig.update_layout(yaxis_title="", xaxis_title="")
                st.plotly_chart(styled_chart_layout(fig, 250), use_container_width=True)
        with c2:
            with st.container(border=True):
                st.markdown("#### Maintenance Recommendation")
                recs = []
                if m["vibration"] > 50:
                    recs.append(("Inspect bearing and lubrication system", "High"))
                if m["pressure"] > 110:
                    recs.append(("Check and replace worn components", "High"))
                if m["error_count"] > 2:
                    recs.append(("Investigate recurring error codes", "Medium"))
                if not recs:
                    recs.append(("No immediate action needed — continue routine checks", "Low"))
                for text, pri in recs:
                    st.markdown(f'<div class="alert-row"><div style="flex:1">{text}</div>{pill(pri, pri)}</div>',
                                unsafe_allow_html=True)
                st.button("Schedule Maintenance", use_container_width=True)

    with tab_sensors:
        with st.container(border=True):
            st.markdown("#### Live Sensor Snapshot")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Voltage", f"{m['volt']} V")
            s2.metric("Rotation", f"{m['rotate']} rpm")
            s3.metric("Pressure", f"{m['pressure']} psi")
            s4.metric("Vibration", f"{m['vibration']} mm/s")
        with st.container(border=True):
            st.markdown("#### Simulated Sensor History (last 20 readings)")
            hist = pd.DataFrame({
                "reading": range(1, 21),
                "vibration": np.clip(m["vibration"] + np.random.normal(0, 4, 20), 0, None),
                "pressure": np.clip(m["pressure"] + np.random.normal(0, 6, 20), 0, None),
            })
            fig = px.line(hist, x="reading", y=["vibration", "pressure"])
            st.plotly_chart(styled_chart_layout(fig, 280), use_container_width=True)

    with tab_ai:
        with st.container(border=True):
            st.markdown("#### Model Prediction Breakdown")
            st.write(f"**Predicted failure probability:** {m['fail_prob']}%")
            st.write(f"**Predicted risk level:** {m['risk']}")
            st.write(f"**Predicted status:** {m['status']}")
            st.caption("Prediction generated by your trained Random Forest model "
                       "via `preprocessor.transform()` → `model.predict_proba()`."
                       if model is not None else
                       "⚠️ Using fallback formula — load your .pkl files to see real model output here.")

    with tab_maint:
        with st.container(border=True):
            st.markdown("#### Maintenance Log")
            log = pd.DataFrame({
                "Date": [m["last_maint"]],
                "Type": ["Routine Inspection"],
                "Notes": ["No components replaced"],
            })
            st.dataframe(log, use_container_width=True, hide_index=True)

    with tab_history:
        with st.container(border=True):
            st.markdown("#### Health Score History")
            hist2 = pd.DataFrame({
                "day": pd.date_range(end=datetime.now(), periods=14),
                "health": np.clip(m["health"] + np.random.normal(0, 5, 14).cumsum() * 0.3, 0, 100),
            })
            fig = px.line(hist2, x="day", y="health")
            fig.update_traces(line_color=ACCENT)
            st.plotly_chart(styled_chart_layout(fig, 280), use_container_width=True)


# =========================================================
# PAGE: ANALYTICS
# =========================================================
elif page == "Analytics":
    st.title("Analytics")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        with st.container(border=True):
            st.markdown("#### Health Score Distribution")
            bins = pd.cut(fleet["health"], bins=[0, 25, 50, 75, 100], labels=["0-25", "25-50", "50-75", "75-100"])
            fig = px.pie(values=bins.value_counts().values, names=bins.value_counts().index, hole=0.5)
            st.plotly_chart(styled_chart_layout(fig, 260), use_container_width=True)
    with c2:
        with st.container(border=True):
            st.markdown("#### Failure Prob by Model")
            by_model = fleet.groupby("model")["fail_prob"].mean().reset_index()
            fig = px.bar(by_model, x="model", y="fail_prob")
            fig.update_traces(marker_color="#3b82f6")
            st.plotly_chart(styled_chart_layout(fig, 260), use_container_width=True)
    with c3:
        with st.container(border=True):
            st.markdown("#### Average Sensor Readings")
            sens = fleet[["volt", "rotate", "pressure", "vibration"]].mean().reset_index()
            sens.columns = ["sensor", "avg"]
            fig = px.bar(sens, x="sensor", y="avg")
            fig.update_traces(marker_color="#8b5cf6")
            st.plotly_chart(styled_chart_layout(fig, 260), use_container_width=True)

    with st.container(border=True):
        st.markdown("#### Sensor Correlation Heatmap")
        corr = fleet[["volt", "rotate", "pressure", "vibration", "age", "fail_prob"]].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
        st.plotly_chart(styled_chart_layout(fig, 350), use_container_width=True)


# =========================================================
# PAGE: ALERTS
# =========================================================
elif page == "Alerts":
    st.title("Alerts")
    with st.container(border=True):
        colf1, colf2 = st.columns(2, gap="medium")
        risk_filter = colf1.selectbox("Risk Level", ["All"] + list(COLORS.keys()))
        status_filter = colf2.selectbox("Status", ["All", "New", "In Progress", "Resolved"])

        view = alerts.copy()
        if risk_filter != "All":
            view = view[view["risk"] == risk_filter]
        if status_filter != "All":
            view = view[view["status"] == status_filter]

        for _, a in view.iterrows():
            st.markdown(
                f'<div class="alert-row"><div class="alert-dot" style="background:{COLORS[a["risk"]]}"></div>'
                f'<div style="flex:1"><b>{a["message"]}</b> — {a["machineID"]}<br>'
                f'<span style="color:#8ea0bf;font-size:11px">{a["time"]}</span></div>'
                f'{pill(a["risk"], a["risk"])}&nbsp;&nbsp;{a["status"]}</div>', unsafe_allow_html=True)


# =========================================================
# PAGE: REPORTS
# =========================================================
elif page == "Reports":
    st.title("Reports")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        with st.container(border=True):
            st.markdown("#### Machine Report")
            st.caption("Generate a detailed report for a specific machine.")
            st.button("Generate Machine Report")
    with c2:
        with st.container(border=True):
            st.markdown("#### Monthly Report")
            st.caption("Generate the monthly fleet performance report.")
            st.button("Generate Monthly Report")
    with c3:
        with st.container(border=True):
            st.markdown("#### Custom Report")
            st.caption("Generate a report for a custom date range.")
            st.button("Generate Custom Report")

    with st.container(border=True):
        st.markdown("#### Export Data")
        st.download_button("⬇ Export Fleet Data (CSV)", fleet.to_csv(index=False), "fleet_data.csv", "text/csv")
        st.download_button("⬇ Export Alerts (CSV)", alerts.to_csv(index=False), "alerts.csv", "text/csv")


# =========================================================
# PAGE: SETTINGS / ABOUT
# =========================================================
elif page == "Settings / About":
    st.title("Settings / About")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        with st.container(border=True):
            st.markdown("#### About Project")
            st.write("NEXMAINT is an AI-powered predictive maintenance system that uses "
                     "machine learning to predict machine failures and reduce downtime.")
            st.caption("Version: 1.0.0")
    with c2:
        with st.container(border=True):
            st.markdown("#### Dataset Information")
            st.write("**Dataset:** Microsoft Azure Predictive Maintenance")
            st.write("**Total Records:** 877,209")
            st.write("**Total Features:** 12")
            st.write("**Target Column:** failure (0 = No Failure, 1 = Failure)")
    with c3:
        with st.container(border=True):
            st.markdown("#### Model Information")
            st.write("**Algorithm:** Random Forest Classifier")
            st.write("**Accuracy:** 92.4% &nbsp; **ROC-AUC:** 0.96")
            st.write("**Precision:** 0.91 &nbsp; **Recall:** 0.90 &nbsp; **F1:** 0.90")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True):
            st.markdown("#### Model Parameters")
            st.write("n_estimators: 200")
            st.write("max_depth: None")
            st.write("min_samples_split: 2")
            st.write("min_samples_leaf: 1")
            st.write("max_features: sqrt")
            st.write("random_state: 42")
    with c2:
        with st.container(border=True):
            st.markdown("#### Developer Information")
            st.write("**Role:** B.Tech CSE Student")
            st.write("**Project Type:** Final Year Project")
