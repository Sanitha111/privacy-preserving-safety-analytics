# dashboard/app.py — Ghost-Vision Dashboard (Review 3)
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import initialize_db, get_connection, get_embedding_connection
from agents.skeleton_extractor import SkeletonExtractorAgent
from agents.graph_builder import GraphBuilderAgent
from agents.action_recognizer import ActionRecognizerAgent
from agents.privacy_manager import PrivacyManagerAgent
from agents.alert_generator import AlertGeneratorAgent
from config import ACTIONS

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Ghost-Vision | Safety AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

* { box-sizing: border-box; }

.header {
    background: linear-gradient(135deg, #050d1a, #0a1628, #0d2137);
    border: 1px solid #00ff88;
    border-radius: 16px;
    padding: 28px 36px;
    text-align: center;
    margin-bottom: 24px;
    box-shadow: 0 0 40px rgba(0,255,136,0.15), inset 0 1px 0 rgba(255,255,255,0.05);
}
.header h1 {
    font-family: 'Orbitron', monospace;
    color: #00ff88;
    font-size: 2.8em;
    margin: 0 0 6px 0;
    letter-spacing: 4px;
    text-shadow: 0 0 30px rgba(0,255,136,0.6);
}
.header p {
    color: #7ab8d4;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.9em;
    margin: 4px 0;
}
.header .badges {
    color: #f0c040;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8em;
    margin-top: 8px;
}

.status-normal {
    background: linear-gradient(135deg, #002d1a, #003d22);
    border: 2px solid #00ff88;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 0 20px rgba(0,255,136,0.2);
}
.status-normal .label {
    font-family: 'Share Tech Mono', monospace;
    color: #7ab8d4;
    font-size: 0.85em;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.status-normal .action {
    font-family: 'Orbitron', monospace;
    color: #00ff88;
    font-size: 2em;
    font-weight: 700;
    margin: 0;
}
.status-normal .confidence {
    color: #7ab8d4;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1em;
    margin-top: 4px;
}

.status-danger {
    background: linear-gradient(135deg, #2d0000, #3d0000);
    border: 2px solid #ff3333;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 0 30px rgba(255,51,51,0.4);
    animation: pulse-red 1.5s infinite;
}
.status-danger .label {
    font-family: 'Share Tech Mono', monospace;
    color: #ff8888;
    font-size: 0.85em;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.status-danger .action {
    font-family: 'Orbitron', monospace;
    color: #ff3333;
    font-size: 2em;
    font-weight: 900;
    margin: 0;
}
.status-danger .confidence {
    color: #ff8888;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1em;
    margin-top: 4px;
}

@keyframes pulse-red {
    0% { box-shadow: 0 0 20px rgba(255,51,51,0.3); }
    50% { box-shadow: 0 0 50px rgba(255,51,51,0.7); }
    100% { box-shadow: 0 0 20px rgba(255,51,51,0.3); }
}

.metric-card {
    background: #050d1a;
    border: 1px solid #1a3a5c;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-card .metric-label {
    font-family: 'Share Tech Mono', monospace;
    color: #5a8aaa;
    font-size: 0.75em;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.metric-card .metric-value {
    font-family: 'Orbitron', monospace;
    color: #00ff88;
    font-size: 2.2em;
    font-weight: 700;
    line-height: 1.2;
}
.metric-card .metric-value.danger { color: #ff3333; }
.metric-card .metric-sub {
    font-family: 'Share Tech Mono', monospace;
    color: #3a6a8a;
    font-size: 0.7em;
    margin-top: 4px;
}

.alert-item {
    background: linear-gradient(135deg, #1a0000, #2d0000);
    border-left: 4px solid #ff3333;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-family: 'Share Tech Mono', monospace;
}
.alert-item .alert-action { color: #ff6666; font-weight: bold; font-size: 1em; }
.alert-item .alert-meta { color: #885555; font-size: 0.8em; margin-top: 4px; }

.safe-item {
    background: #050d1a;
    border-left: 4px solid #1a4a2a;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-family: 'Share Tech Mono', monospace;
}
.safe-item .safe-action { color: #4a8a5a; font-size: 0.95em; }
.safe-item .safe-meta { color: #2a5a3a; font-size: 0.8em; margin-top: 4px; }

.privacy-box {
    background: #050d1a;
    border: 1px solid #1a3a5c;
    border-radius: 12px;
    padding: 20px;
    font-family: 'Share Tech Mono', monospace;
    color: #7ab8d4;
    margin: 8px 0;
}
.privacy-box h4 { color: #00ff88; font-family: 'Orbitron', monospace; margin: 0 0 8px 0; font-size: 0.9em; }

.section-title {
    font-family: 'Orbitron', monospace;
    color: #7ab8d4;
    font-size: 1em;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 20px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #1a3a5c;
}

.erasure-success {
    background: linear-gradient(135deg, #002d1a, #003d22);
    border: 2px solid #00ff88;
    border-radius: 12px;
    padding: 20px;
    font-family: 'Share Tech Mono', monospace;
    color: #7ab8d4;
}
.erasure-success h3 { color: #00ff88; font-family: 'Orbitron', monospace; margin: 0 0 12px 0; }

.stButton > button {
    background: linear-gradient(135deg, #00ff88, #00cc66) !important;
    color: #050d1a !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Orbitron', monospace !important;
    letter-spacing: 1px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00ffaa, #00ff88) !important;
    box-shadow: 0 0 20px rgba(0,255,136,0.4) !important;
}

body, .main { background-color: #020810 !important; }
.stTabs [data-baseweb="tab"] {
    font-family: 'Share Tech Mono', monospace !important;
    color: #5a8aaa !important;
}
.stTabs [aria-selected="true"] {
    color: #00ff88 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Initialize ───────────────────────────────────────────────
initialize_db()

@st.cache_resource
def load_agents():
    extractor     = SkeletonExtractorAgent()
    graph_agent   = GraphBuilderAgent()
    recognizer    = ActionRecognizerAgent()
    privacy_agent = PrivacyManagerAgent()
    alert_agent   = AlertGeneratorAgent()
    return extractor, graph_agent, recognizer, privacy_agent, alert_agent

extractor, graph_agent, recognizer, privacy_agent, alert_agent = load_agents()

@st.cache_resource
def get_adjacency():
    from utils.ntu_loader import build_ntu_adjacency
    return build_ntu_adjacency()

A = get_adjacency()

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1>👁️ GHOST-VISION</h1>
    <p>Auto-Anonymizing Safety Analytics with On-Demand Privacy Erasure</p>
    <div class="badges">⚡ DPDP Act 2026 Compliant &nbsp;|&nbsp; Privacy by Design &nbsp;|&nbsp; Zero Identity Storage</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h3 style='font-family:Orbitron;color:#00ff88;text-align:center;'>CONTROLS</h3>", unsafe_allow_html=True)

    environment = st.selectbox("🏥 Environment", ["Hospital", "Factory", "General"])

    st.divider()

    # ── Live Webcam ──
    if st.button("📷 Start Live Webcam", use_container_width=True, type="primary"):
        with st.spinner("Webcam running... press Q in the window to stop"):
            sequences = extractor.extract_from_webcam(duration_seconds=20, environment=environment)
            if sequences:
                webcam_results = []
                for seq in sequences:
                    seq_25 = seq[:, :25, :]
                    result = recognizer.predict(seq_25, A)
                    anon_id, _ = privacy_agent.register_person(
                        person_id="webcam_person_1",
                        environment=environment
                    )
                    alert_agent.process_detection(result, anon_id, environment)
                    webcam_results.append(result)
                st.session_state["webcam_results"] = webcam_results
                st.success(f"✅ {len(sequences)} sequences processed!")
            else:
                st.warning("No sequences captured.")

    # ── Demo ──
    if st.button("🧪 Run Demo (No Webcam)", use_container_width=True):
        with st.spinner("Running demo..."):
            for action_name in ["normal", "fall", "normal", "motionless", "normal"]:
                seq     = extractor.generate_demo_skeleton_sequence(action_name)
                seq_25  = seq[:, :25, :]
                result  = recognizer.predict(seq_25, A)
                anon_id, _ = privacy_agent.register_person(
                    person_id=f"demo_{action_name}",
                    environment=environment
                )
                alert_agent.process_detection(result, anon_id, environment)
        st.success("✅ Demo complete!")

    if st.button("🔄 Clear All Data", use_container_width=True):
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM skeleton_events")
        cursor.execute("DELETE FROM alerts")
        cursor.execute("DELETE FROM erasure_requests")
        conn.commit()
        conn.close()
        emb_conn = get_embedding_connection()
        emb_cursor = emb_conn.cursor()
        emb_cursor.execute("DELETE FROM face_embeddings")
        emb_conn.commit()
        emb_conn.close()
        if "webcam_results" in st.session_state:
            del st.session_state["webcam_results"]
        st.success("✅ Cleared!")

    st.divider()
    st.markdown("<p style='font-family:Share Tech Mono;color:#3a6a8a;font-size:0.75em;text-align:center;'>MSc Data Science | VIT AP<br>Krify Technologies Internship<br>Reg: 24MSD7014</p>", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔴  LIVE MONITOR", "📊  ANALYTICS", "🔐  PRIVACY & ERASURE"])

# ════════════════════════════════════════════
# TAB 1 — LIVE MONITOR
# ════════════════════════════════════════════
with tab1:

    # ── Top metrics ──
    privacy_stats = privacy_agent.get_privacy_stats()
    event_stats   = alert_agent.get_event_statistics()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Persons Monitored</div>
            <div class="metric-value">{privacy_stats.get('total_persons_registered', 0)}</div>
            <div class="metric-sub">Today</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Events Detected</div>
            <div class="metric-value">{event_stats.get('total_events', 0)}</div>
            <div class="metric-sub">Total sequences</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        danger_count = event_stats.get('dangerous_events', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Alerts Triggered</div>
            <div class="metric-value {'danger' if danger_count > 0 else ''}">{danger_count}</div>
            <div class="metric-sub">Falls & Motionless</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Raw Videos Stored</div>
            <div class="metric-value">0 ✅</div>
            <div class="metric-sub">Privacy by Design</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Latest Detection Results</div>", unsafe_allow_html=True)

    # ── Webcam results ──
    if "webcam_results" in st.session_state and st.session_state["webcam_results"]:
        results = st.session_state["webcam_results"]
        last    = results[-1]

        # Big status card for last detection
        if last["is_dangerous"]:
            st.markdown(f"""
            <div class="status-danger">
                <div class="label">⚠️ ALERT — IMMEDIATE ATTENTION NEEDED</div>
                <div class="action">🚨 {last['action'].upper()} DETECTED</div>
                <div class="confidence">Confidence: {last['confidence']:.1%} | Environment: Hospital</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-normal">
                <div class="label">● MONITORING ACTIVE</div>
                <div class="action">✅ {last['action'].upper()}</div>
                <div class="confidence">Confidence: {last['confidence']:.1%} | Environment: Hospital</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Sequence by Sequence Results</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        for i, r in enumerate(results):
            with col1 if i % 2 == 0 else col2:
                if r["is_dangerous"]:
                    st.markdown(f"""
                    <div class="alert-item">
                        <div class="alert-action">🚨 Seq {i+1}: {r['action']}</div>
                        <div class="alert-meta">Confidence: {r['confidence']:.1%} — ALERT SENT</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-item">
                        <div class="safe-action">✅ Seq {i+1}: {r['action']}</div>
                        <div class="safe-meta">Confidence: {r['confidence']:.1%} — Normal</div>
                    </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="privacy-box" style="text-align:center; padding:40px;">
            <h4>NO ACTIVE SESSION</h4>
            <p>Click "Start Live Webcam" in the sidebar to begin monitoring.<br>
            Or click "Run Demo" to see a simulated detection.</p>
        </div>""", unsafe_allow_html=True)

    # ── Recent alerts from DB ──
    st.markdown("<div class='section-title'>Recent Alerts from Database</div>", unsafe_allow_html=True)
    recent_alerts = alert_agent.get_recent_alerts(5)
    if len(recent_alerts) > 0:
        for _, alert in recent_alerts.iterrows():
            action     = alert.get("action", "Unknown")
            confidence = alert.get("confidence", 0)
            env        = alert.get("environment", "Unknown")
            ts         = str(alert.get("timestamp", ""))[:19]
            if action not in ["Normal", "Sit Down"]:
                st.markdown(f"""
                <div class="alert-item">
                    <div class="alert-action">🚨 {action}</div>
                    <div class="alert-meta">Confidence: {confidence:.1%} | {env} | {ts}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="safe-item">
                    <div class="safe-action">✅ {action}</div>
                    <div class="safe-meta">Confidence: {confidence:.1%} | {env} | {ts}</div>
                </div>""", unsafe_allow_html=True)
    else:
        st.markdown("<div class='privacy-box'>No events yet — run webcam or demo first.</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>Action Distribution</div>", unsafe_allow_html=True)

    action_counts = event_stats.get("action_counts", {})
    if action_counts:
        col1, col2 = st.columns(2)
        with col1:
            colors = {
                "Normal": "#00ff88", "Fall": "#ff3333",
                "Motionless": "#ff8800", "Pre-Fall Risk": "#ffcc00", "Sit Down": "#4488ff"
            }
            fig = go.Figure(go.Pie(
                labels=list(action_counts.keys()),
                values=list(action_counts.values()),
                marker_colors=[colors.get(k, "#888") for k in action_counts.keys()],
                hole=0.5,
                textfont=dict(family="Share Tech Mono", color="white")
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                height=320,
                margin=dict(t=20, b=20),
                legend=dict(font=dict(family="Share Tech Mono", color="#7ab8d4"))
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("<div class='section-title'>Model Performance</div>", unsafe_allow_html=True)
            models = ["ST-GCN\n(Fall Detection)", "PreFall LSTM\n(Risk Prediction)", "Severity CNN\n(Fall Grading)"]
            accs   = [94.26, 96.25, 92.06]
            fig2 = go.Figure(go.Bar(
                x=models,
                y=accs,
                marker_color=["#00ff88", "#00ccff", "#ffcc00"],
                text=[f"{a}%" for a in accs],
                textposition="outside",
                textfont=dict(family="Share Tech Mono", color="white")
            ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(5,13,26,0.8)",
                font_color="white",
                font_family="Share Tech Mono",
                yaxis=dict(range=[0, 105], gridcolor="#1a3a5c"),
                xaxis=dict(gridcolor="#1a3a5c"),
                height=320,
                margin=dict(t=20, b=20)
            )
            st.plotly_chart(fig2, use_container_width=True)

    else:
        st.markdown("<div class='privacy-box'>Run webcam or demo first to see analytics.</div>", unsafe_allow_html=True)

    # Storage comparison
    st.markdown("<div class='section-title'>Storage: Ghost-Vision vs Traditional CCTV</div>", unsafe_allow_html=True)
    fig3 = go.Figure(go.Bar(
        x=["Traditional CCTV\n(1 hour)", "Ghost-Vision\n(1 hour)"],
        y=[2048, 45],
        marker_color=["#ff3333", "#00ff88"],
        text=["2048 MB", "45 MB"],
        textposition="outside",
        textfont=dict(family="Share Tech Mono", color="white")
    ))
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,13,26,0.8)",
        font_color="white",
        font_family="Share Tech Mono",
        yaxis=dict(title="Storage (MB)", gridcolor="#1a3a5c"),
        height=280,
        margin=dict(t=20, b=20),
        annotations=[dict(
            x=1, y=200,
            text="97.8% less storage ✅",
            showarrow=False,
            font=dict(color="#00ff88", family="Share Tech Mono", size=14)
        )]
    )
    st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════
# TAB 3 — PRIVACY & ERASURE
# ════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>Privacy Statistics</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Raw Video Stored</div>
            <div class="metric-value">0 MB</div>
            <div class="metric-sub">Always zero</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Face Photos Stored</div>
            <div class="metric-value">0</div>
            <div class="metric-sub">Math vectors only</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        erasure_count = privacy_stats.get('total_erasure_requests', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Erasure Requests</div>
            <div class="metric-value">{erasure_count}</div>
            <div class="metric-sub">DPDP Section 12</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>How Privacy Works</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="privacy-box">
        <h4>PRIVACY PIPELINE</h4>
        <p>📹 Camera captures video &nbsp;→&nbsp; ⚡ Raw video DISCARDED immediately<br>
        🦴 33 skeleton joints extracted &nbsp;→&nbsp; 🔢 Stored as X,Y,Z numbers only<br>
        👤 Face converted to 128 numbers &nbsp;→&nbsp; 🚫 Face photo NEVER stored<br>
        🆔 Anonymous ID assigned &nbsp;→&nbsp; ✅ No name, no face, no identity</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>DPDP Act 2026 — Right to Erasure (Section 12)</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="privacy-box">
        <h4>HOW ERASURE WORKS</h4>
        <p>
        1. Person provides ONE photo of themselves<br>
        2. System converts photo → 128-number mathematical embedding<br>
        3. System searches database using cosine similarity<br>
        4. ALL matching records permanently deleted<br>
        5. Erasure certificate generated for legal compliance
        </p>
    </div>""", unsafe_allow_html=True)

    person_to_erase = st.selectbox(
        "Select person to erase:",
        ["webcam_person_1", "demo_fall", "demo_normal", "demo_motionless"]
    )

    if st.button("🗑️ Exercise Right to Erasure", type="primary"):
        with st.spinner("Processing DPDP erasure request..."):
            result = privacy_agent.exercise_right_to_erasure(person_id=person_to_erase)

        if result["success"]:
            st.markdown(f"""
            <div class="erasure-success">
                <h3>✅ ERASURE COMPLETE — DPDP ACT COMPLIANT</h3>
                <p><strong>Request ID:</strong> {result['request_id']}</p>
                <p><strong>Person:</strong> {person_to_erase} → Permanently removed</p>
                <p><strong>Records Deleted:</strong> {result['records_deleted']}</p>
                <p><strong>Legal Status:</strong> DPDP Act 2026 Section 12 ✅</p>
                <p style='color:#3a6a4a;font-size:0.85em;'>"{result['message']}"</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.warning(f"⚠️ {result['message']} — Run demo or webcam first to register persons.")

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.markdown("""
<p style='text-align:center;font-family:Share Tech Mono;color:#1a3a5c;font-size:0.75em;'>
👁️ GHOST-VISION &nbsp;|&nbsp; S. Anitha &nbsp;|&nbsp; 24MSD7014 &nbsp;|&nbsp; MSc Data Science &nbsp;|&nbsp; VIT AP &nbsp;|&nbsp; Krify Technologies
</p>""", unsafe_allow_html=True)