from pathlib import Path
import hashlib
import json

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "threat_model.joblib"
DATA_PATH = ROOT / "data" / "network_traffic.csv"
PREDICTIONS_PATH = ROOT / "outputs" / "threat_alerts.csv"
EVALUATION_PATH = ROOT / "outputs" / "evaluation.json"
CASE_FILE = ROOT / "outputs" / "incident_cases.csv"

st.set_page_config(
    page_title="CyberShield AI | SOC Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- DARK SOC THEME ----------------
st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp { background: #070B12; color: #E6EDF7; }
    [data-testid="stHeader"] { background: rgba(7,11,18,.96); }
    [data-testid="stSidebar"] { background: #0B1220; border-right: 1px solid #203047; }
    .block-container { padding-top: 1.3rem; padding-bottom: 2rem; max-width: 1600px; }
    h1,h2,h3,h4 { color: #F1F5F9 !important; }
    p, label, .stMarkdown, [data-testid="stCaptionContainer"] { color: #C5D0E0; }
    [data-testid="stMetric"] { background:#101A2A; border:1px solid #26364B; padding:16px; border-radius:12px; }
    [data-testid="stMetricLabel"] { color:#91A4BC !important; }
    [data-testid="stMetricValue"] { color:#2DD4BF !important; }
    div[data-testid="stDataFrame"] { border:1px solid #26364B; border-radius:10px; }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea { background:#101A2A; color:#E6EDF7; border-color:#334155; }
    .stButton > button { background:#123C46; color:#F0FDFA; border:1px solid #2DD4BF; border-radius:8px; font-weight:600; }
    .stButton > button:hover { background:#155E63; color:white; border-color:#5EEAD4; }
    div[data-testid="stAlert"] { border-radius:10px; }
    hr { border-color:#26364B; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- HEADER ----------------
st.title("🛡️ CyberShield AI")
st.caption("SOC COMMAND CENTER  /  AI-POWERED NETWORK THREAT DETECTION")
st.info(
    "DEMO ENVIRONMENT — This dashboard analyses synthetic network traffic only. "
    "It is not connected to a live network and does not block real devices."
)

# ---------------- FILE CHECKS / LOADERS ----------------
required_files = [MODEL_PATH, DATA_PATH, PREDICTIONS_PATH, EVALUATION_PATH]
missing = [p.name for p in required_files if not p.exists()]
if missing:
    st.error("Required files missing: " + ", ".join(missing))
    st.info("From the project folder run: `python src/train_model.py` then `python src/evaluate_model.py`.")
    st.stop()

@st.cache_data
def load_alerts(path, modified_time):
    return pd.read_csv(path)

@st.cache_data
def load_metrics(path, modified_time):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

alerts = load_alerts(str(PREDICTIONS_PATH), PREDICTIONS_PATH.stat().st_mtime).copy()
metrics = load_metrics(str(EVALUATION_PATH), EVALUATION_PATH.stat().st_mtime)

if "predicted_label" not in alerts.columns:
    st.error("The predictions file must contain a 'predicted_label' column.")
    st.stop()

alerts["predicted_label"] = alerts["predicted_label"].fillna("Unknown").astype(str)
if "protocol" not in alerts.columns:
    alerts["protocol"] = "Unknown"
if "status" not in alerts.columns:
    alerts["status"] = np.where(alerts["predicted_label"].str.lower().eq("normal"), "NORMAL", "SUSPICIOUS")

def stable_incident_id(row):
    """Create an ID based on event features, not current row position."""
    keys = [
        "duration", "src_bytes", "dst_bytes", "failed_logins",
        "connection_count", "unique_ports", "protocol", "predicted_label"
    ]
    values = [str(row.get(k, "")) for k in keys]
    digest = hashlib.sha1("|".join(values).encode("utf-8")).hexdigest()[:10].upper()
    return f"INC-{digest}"

alerts["incident_id"] = alerts.apply(stable_incident_id, axis=1)

def risk_score(row):
    """Transparent demo heuristic; not a calibrated real-world probability."""
    label = str(row.get("predicted_label", "Normal")).lower()
    score = 18.0 if label == "normal" else 58.0
    try:
        failed = float(row.get("failed_logins", 0) or 0)
        connections = float(row.get("connection_count", 0) or 0)
        ports = float(row.get("unique_ports", 0) or 0)
        src_bytes = float(row.get("src_bytes", 0) or 0)
        if failed >= 5:
            score += 18
        if connections >= 40:
            score += 12
        if ports >= 15:
            score += 10
        if src_bytes >= 100000:
            score += 5
    except (ValueError, TypeError):
        pass
    return int(min(99, max(5, score)))

def investigation_reason(row):
    label = str(row.get("predicted_label", "Unknown"))
    reasons = []
    try:
        if float(row.get("failed_logins", 0) or 0) >= 5:
            reasons.append("multiple failed login attempts")
        if float(row.get("connection_count", 0) or 0) >= 40:
            reasons.append("high connection volume")
        if float(row.get("unique_ports", 0) or 0) >= 15:
            reasons.append("many unique destination ports")
        if float(row.get("src_bytes", 0) or 0) >= 100000:
            reasons.append("high source-byte volume")
    except (ValueError, TypeError):
        pass
    if label.lower() == "normal":
        return "Model classified this event as Normal; still a synthetic example."
    if reasons:
        return f"Model predicted {label}; observed indicators: " + ", ".join(reasons) + "."
    return f"Model predicted {label}. Review the displayed features; this demo has no validated forensic explanation."

alerts["risk_score"] = alerts.apply(risk_score, axis=1)
alerts["investigation_reason"] = alerts.apply(investigation_reason, axis=1)
alerts["incident_severity"] = pd.cut(
    alerts["risk_score"], bins=[0, 39, 69, 100],
    labels=["LOW", "MEDIUM", "HIGH"], include_lowest=True
).astype(str)

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.title("⚙️ SOC Controls")
st.sidebar.caption("Synthetic traffic exploration")
categories = sorted(alerts["predicted_label"].unique().tolist())
selected_categories = st.sidebar.multiselect("Attack / traffic category", categories, default=categories)
status_filter = st.sidebar.radio("Traffic status", ["All", "Suspicious only", "Normal only"])
search_text = st.sidebar.text_input("Search protocol", "")

filtered = alerts[alerts["predicted_label"].isin(selected_categories)].copy()
if status_filter == "Suspicious only":
    filtered = filtered[filtered["predicted_label"].str.lower() != "normal"]
elif status_filter == "Normal only":
    filtered = filtered[filtered["predicted_label"].str.lower() == "normal"]
if search_text:
    filtered = filtered[filtered["protocol"].astype(str).str.contains(search_text, case=False, na=False)]

# ---------------- TOP KPIs ----------------
total_records = len(alerts)
suspicious = alerts[alerts["predicted_label"].str.lower() != "normal"]
normal_records = total_records - len(suspicious)
accuracy = metrics.get("accuracy", 0)
try:
    accuracy = float(accuracy)
except (TypeError, ValueError):
    accuracy = 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Traffic Records", f"{total_records:,}")
k2.metric("Predicted Threats", f"{len(suspicious):,}")
k3.metric("Normal Traffic", f"{normal_records:,}")
k4.metric("Held-out Test Accuracy", f"{accuracy:.1%}")
st.caption("Threat counts are predictions on synthetic demo data. Accuracy is from the held-out test split.")

# ---------------- SUMMARY CHARTS ----------------
st.divider()
chart_left, chart_right = st.columns(2)
dark_layout = dict(
    paper_bgcolor="#0B1220",
    plot_bgcolor="#0B1220",
    font=dict(color="#D8E2F0"),
    margin=dict(l=12, r=12, t=45, b=12),
)
with chart_left:
    st.subheader("📊 Predicted Traffic Categories")
    dist = alerts["predicted_label"].value_counts().rename_axis("Category").reset_index(name="Records")
    fig = px.bar(
        dist, x="Category", y="Records", color="Category",
        color_discrete_sequence=["#2DD4BF", "#F97316", "#EF4444", "#60A5FA", "#A78BFA"],
    )
    fig.update_layout(**dark_layout, showlegend=False)
    fig.update_xaxes(gridcolor="#223047")
    fig.update_yaxes(gridcolor="#223047")
    st.plotly_chart(fig, use_container_width=True)
with chart_right:
    st.subheader("🚨 Normal vs Suspicious")
    summary = pd.DataFrame({"Status": ["Normal", "Suspicious"], "Records": [normal_records, len(suspicious)]})
    fig2 = px.pie(
        summary, names="Status", values="Records", hole=0.58,
        color="Status", color_discrete_map={"Normal": "#2DD4BF", "Suspicious": "#EF4444"},
    )
    fig2.update_layout(**dark_layout, legend=dict(font=dict(color="#D8E2F0")))
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- NETWORK THREAT MAP ----------------
st.divider()
st.header("🌐 Network Threat Map")
st.caption("Interactive synthetic topology — click a node to select an event for investigation.")

map_source = filtered.copy()
if map_source.empty:
    st.info("No events match the current sidebar filters. Widen the filters to show the network map.")
else:
    # Keep graph responsive and easy to read.
    node_limit = min(60, len(map_source))
    if len(map_source) > node_limit:
        map_source = map_source.sort_values("risk_score", ascending=False).head(node_limit).copy()
    map_source = map_source.drop_duplicates("incident_id").reset_index(drop=True)

    graph = nx.Graph()
    graph.add_node("SOC", kind="hub", label="SOC")
    for _, row in map_source.iterrows():
        iid = row["incident_id"]
        graph.add_node(iid, kind="event", label=iid, row=row.to_dict())
        graph.add_edge("SOC", iid)
        # A small number of peer links create a network-like topology.
    event_ids = map_source["incident_id"].tolist()
    for i in range(0, max(0, len(event_ids) - 1), 3):
        graph.add_edge(event_ids[i], event_ids[i + 1])

    pos = nx.spring_layout(graph, seed=21, k=0.9, iterations=60)
    edge_x, edge_y = [], []
    for source, target in graph.edges():
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1, color="#284257"),
        hoverinfo="none",
    )

    node_x, node_y, node_text, node_color, node_size, node_ids = [], [], [], [], [], []
    for node, attrs in graph.nodes(data=True):
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        node_ids.append(node)
        if node == "SOC":
            node_text.append("SOC Command Center")
            node_color.append("#2DD4BF")
            node_size.append(27)
        else:
            row = attrs["row"]
            label = str(row.get("predicted_label", "Unknown"))
            score = int(row.get("risk_score", 0))
            node_text.append(
                f"{node}<br>Attack: {label}<br>Risk score: {score}/100"
                f"<br>Severity: {row.get('incident_severity', 'UNKNOWN')}"
            )
            node_color.append("#EF4444" if score >= 70 else "#F97316" if score >= 40 else "#2DD4BF")
            node_size.append(17 if score >= 70 else 14)
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=["SOC" if n == "SOC" else n[-4:] for n in node_ids],
        textposition="top center",
        textfont=dict(color="#C8D7E8", size=9),
        customdata=node_ids,
        hovertext=node_text,
        hovertemplate="%{hovertext}<extra></extra>",
        marker=dict(size=node_size, color=node_color, line=dict(width=1.5, color="#D5E6F5")),
    )
    netfig = go.Figure(data=[edge_trace, node_trace])
    netfig.update_layout(
        **dark_layout,
        height=560,
        showlegend=False,
        clickmode="event+select",
        dragmode="pan",
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False),
     
    )
    st.plotly_chart(netfig, use_container_width=True, key="network_threat_map")

    # Streamlit's standard Plotly chart does not expose point-click events in every version.
    # This selector provides reliable node selection alongside the clickable/hoverable graph.
    selectable_ids = [i for i in event_ids if i in set(map_source["incident_id"])]
    chosen_node = st.selectbox(
        "Select a network node to investigate",
        options=selectable_ids,
        format_func=lambda iid: (
            f"{iid} · {map_source.loc[map_source['incident_id'].eq(iid), 'predicted_label'].iloc[0]} · "
            f"Risk {int(map_source.loc[map_source['incident_id'].eq(iid), 'risk_score'].iloc[0])}/100"
        ),
        key="selected_network_node",
    )
    st.session_state["active_incident_id"] = chosen_node

# ---------------- MODEL EVALUATION ----------------
st.divider()
st.subheader("🧠 Model Evaluation")
st.caption("Held-out test-set classification report from model training.")
report = metrics.get("classification_report", "")
if report:
    st.code(str(report), language="text")
with st.expander("Show raw evaluation details"):
    st.json(metrics)

# ---------------- NETWORK THREAT INVESTIGATION ----------------
st.divider()
st.header("🔎 Network Threat Investigation")
st.caption("Inspect model predictions, visible features, risk score and rule-based indicator summary.")

threat_source = alerts[alerts["predicted_label"].str.lower() != "normal"].copy()
if threat_source.empty:
    st.success("No suspicious predictions exist in the current synthetic dataset.")
else:
    default_id = st.session_state.get("active_incident_id")
    threat_ids = threat_source["incident_id"].tolist()
    if default_id not in threat_ids:
        default_id = threat_ids[0]
    selected_incident = st.selectbox(
        "Choose suspicious event",
        options=threat_ids,
        index=threat_ids.index(default_id),
        format_func=lambda iid: (
            f"{iid} · {threat_source.loc[threat_source['incident_id'].eq(iid), 'predicted_label'].iloc[0]}"
        ),
        key="investigation_incident",
    )
    event = threat_source.loc[threat_source["incident_id"].eq(selected_incident)].iloc[0]

    r1, r2, r3 = st.columns(3)
    r1.metric("Predicted Attack Type", str(event["predicted_label"]))
    r2.metric("Demo Risk Score", f"{int(event['risk_score'])}/100")
    r3.metric("Severity", str(event["incident_severity"]))

    st.markdown("#### Why was this event flagged?")
    st.warning(str(event["investigation_reason"]))
    st.markdown("#### Event features")
    feature_cols = [
        "duration", "src_bytes", "dst_bytes", "failed_logins",
        "connection_count", "unique_ports", "protocol", "actual_label",
        "predicted_label", "status", "risk_score", "incident_severity",
    ]
    available_features = [c for c in feature_cols if c in event.index]
    st.dataframe(
        pd.DataFrame({"Feature": available_features, "Value": [str(event[c]) for c in available_features]}),
        use_container_width=True,
        hide_index=True,
    )

# ---------------- INCIDENT RESPONSE SIMULATOR ----------------
st.divider()
st.header("🛡️ Incident Response Simulator")
st.caption("Analyst workflow: Open → Acknowledged → Investigating → Resolved. Status changes are simulated only.")
CASE_FILE.parent.mkdir(parents=True, exist_ok=True)

case_columns = ["incident_id", "case_status", "analyst_notes"]
if "incident_cases" not in st.session_state:
    if CASE_FILE.exists():
        try:
            saved_cases = pd.read_csv(CASE_FILE, dtype={"incident_id": str})
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            saved_cases = pd.DataFrame(columns=case_columns)
    else:
        saved_cases = pd.DataFrame(columns=case_columns)
    for col in case_columns:
        if col not in saved_cases.columns:
            saved_cases[col] = "" if col == "analyst_notes" else ("Open" if col == "case_status" else "")
    st.session_state["incident_cases"] = saved_cases[case_columns].copy()

cases = st.session_state["incident_cases"].copy()
for col in case_columns:
    if col not in cases.columns:
        cases[col] = "" if col == "analyst_notes" else ("Open" if col == "case_status" else "")
cases["incident_id"] = cases["incident_id"].astype(str)

# Add new suspicious incidents without resetting saved statuses.
suspicious_rows = alerts[alerts["predicted_label"].str.lower() != "normal"]
known_ids = set(cases["incident_id"])
new_rows = []
for _, row in suspicious_rows.iterrows():
    iid = str(row["incident_id"])
    if iid not in known_ids:
        new_rows.append({"incident_id": iid, "case_status": "Open", "analyst_notes": ""})
if new_rows:
    cases = pd.concat([cases, pd.DataFrame(new_rows)], ignore_index=True)
    cases = cases.drop_duplicates("incident_id", keep="last")
    st.session_state["incident_cases"] = cases
    cases.to_csv(CASE_FILE, index=False)

if suspicious_rows.empty:
    st.info("No suspicious events are available to create incident cases.")
else:
    workflow = suspicious_rows[["incident_id", "predicted_label", "risk_score", "incident_severity"]].merge(
        cases, on="incident_id", how="left"
    )
    workflow["case_status"] = workflow["case_status"].fillna("Open")
    workflow["analyst_notes"] = workflow["analyst_notes"].fillna("")
    open_count = int((workflow["case_status"] == "Open").sum())
    ack_count = int((workflow["case_status"] == "Acknowledged").sum())
    investigating_count = int((workflow["case_status"] == "Investigating").sum())
    resolved_count = int((workflow["case_status"] == "Resolved").sum())
    a, b, c, d = st.columns(4)
    a.metric("Open", open_count)
    b.metric("Acknowledged", ack_count)
    c.metric("Investigating", investigating_count)
    d.metric("Simulated Resolved", resolved_count)

    st.dataframe(
        workflow[["incident_id", "predicted_label", "risk_score", "incident_severity", "case_status"]],
        use_container_width=True,
        hide_index=True,
    )

    active_id = st.session_state.get("active_incident_id")
    options = workflow["incident_id"].tolist()
    if active_id not in options:
        active_id = options[0]
    case_id = st.selectbox(
        "Select incident case",
        options=options,
        index=options.index(active_id),
        format_func=lambda iid: (
            f"{iid} · {workflow.loc[workflow['incident_id'].eq(iid), 'predicted_label'].iloc[0]} · "
            f"{workflow.loc[workflow['incident_id'].eq(iid), 'case_status'].iloc[0]}"
        ),
        key="case_selector",
    )
    case_row = workflow.loc[workflow["incident_id"].eq(case_id)].iloc[0]
    st.write(f"**Current status:** `{case_row['case_status']}`")
    notes = st.text_area(
        "Analyst investigation notes",
        value=str(case_row["analyst_notes"] or ""),
        placeholder="Example: Reviewed simulated failed-login activity; no real system action taken.",
        key=f"notes_{case_id}",
    )

    current_status = str(case_row["case_status"])
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        ack_clicked = st.button("Acknowledge", disabled=current_status != "Open", use_container_width=True)
    with b2:
        investigate_clicked = st.button(
            "Start Investigation",
            disabled=current_status not in ["Open", "Acknowledged"],
            use_container_width=True,
        )
    with b3:
        resolve_clicked = st.button(
            "Mark Simulated Resolved",
            disabled=current_status == "Resolved",
            use_container_width=True,
        )
    with b4:
        save_notes_clicked = st.button("Save Notes", use_container_width=True)

    new_status = None
    if ack_clicked:
        new_status = "Acknowledged"
    elif investigate_clicked:
        new_status = "Investigating"
    elif resolve_clicked:
        new_status = "Resolved"

    if new_status or save_notes_clicked:
        mask = st.session_state["incident_cases"]["incident_id"].astype(str).eq(str(case_id))
        if new_status:
            st.session_state["incident_cases"].loc[mask, "case_status"] = new_status
        st.session_state["incident_cases"].loc[mask, "analyst_notes"] = notes
        st.session_state["incident_cases"].to_csv(CASE_FILE, index=False)
        if new_status:
            st.success(f"{case_id} updated to {new_status}. This is a simulated status only.")
            st.rerun()
        else:
            st.success("Investigation notes saved.")

# ---------------- TRAFFIC TABLE / DOWNLOAD ----------------
st.divider()
st.subheader("📄 Filtered Network Traffic")
st.write(f"Showing **{len(filtered):,}** records after applying sidebar filters.")
display_columns = [
    "incident_id", "duration", "src_bytes", "dst_bytes", "failed_logins",
    "connection_count", "unique_ports", "protocol", "actual_label",
    "predicted_label", "risk_score", "incident_severity", "status",
]
display_columns = [col for col in display_columns if col in filtered.columns]
st.dataframe(filtered[display_columns].head(500), use_container_width=True, hide_index=True)

csv_data = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Filtered Traffic CSV",
    data=csv_data,
    file_name="cybershield_filtered_traffic.csv",
    mime="text/csv",
)

st.caption(
    "CyberShield AI | Student portfolio project | Synthetic data only. "
    "Risk scores are heuristic demo scores, not calibrated probabilities. "
    "Predictions and incident statuses are not validated for production use."
)
