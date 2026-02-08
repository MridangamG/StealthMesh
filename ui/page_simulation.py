"""
Live Attack Simulation Page
"""

import streamlit as st
import time
import random
import plotly.graph_objects as go
from datetime import datetime


def render():
    st.markdown('<h1 class="hero-title">🎮 Live Attack Simulation</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Watch StealthMesh defend against a simulated cyber attack in real-time</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Scenario Selection ──
    scenario = st.selectbox(
        "Select Attack Scenario",
        [
            "🌊 DDoS Flood Attack",
            "🔍 Network Port Scan",
            "🔑 SSH Brute Force",
            "💀 Ransomware Deployment",
            "🕵️ Zero-Day Exploit",
        ],
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        attack_intensity = st.slider("Attack Intensity", 1, 10, 5)
    with col2:
        num_nodes = st.slider("Mesh Nodes", 3, 8, 5)
    with col3:
        speed = st.slider("Simulation Speed", 1, 5, 3)

    if st.button("🚀 Launch Simulation", type="primary", use_container_width=True):
        _run_simulation(scenario, attack_intensity, num_nodes, speed)


def _run_simulation(scenario, intensity, nodes, speed):
    delay = 0.5 / speed

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Status panels
    status_col, log_col = st.columns([1, 2])

    with status_col:
        st.markdown("### 📊 System Status")
        threat_metric = st.empty()
        nodes_metric = st.empty()
        packets_metric = st.empty()
        actions_metric = st.empty()
        phase_display = st.empty()

    with log_col:
        st.markdown("### 📜 Event Log")
        log_container = st.empty()

    # Progress
    progress = st.progress(0.0)

    # Chart area
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns(2)
    threat_chart = chart_col1.empty()
    packet_chart = chart_col2.empty()

    # ── Simulation Phases ──
    attack_name = scenario.split(" ", 1)[1]
    logs = []
    threat_levels = []
    packet_counts = []
    total_steps = 20
    blocked_at = random.randint(8, 12)

    for step in range(total_steps):
        t = step / total_steps
        progress.progress(t)
        timestamp = f"T+{step * 500}ms"

        # Phase 1: Attack begins (0-30%)
        if step < 6:
            phase = "🔴 ATTACK IN PROGRESS"
            threat = min(0.1 + step * 0.12 * (intensity / 5), 0.7)
            pkts = random.randint(100, 500) * intensity

            if step == 0:
                logs.append(f'<span style="color:#e74c3c">[{timestamp}] ⚠️ {attack_name} detected from 10.0.0.{random.randint(1,254)}</span>')
            elif step == 1:
                logs.append(f'<span style="color:#3498db">[{timestamp}] 🤖 ML Model classifying traffic... Confidence: {92+step}%</span>')
            elif step == 2:
                logs.append(f'<span style="color:#f39c12">[{timestamp}] 🔐 Alert encrypted via AES-256-GCM</span>')
            elif step == 3:
                logs.append(f'<span style="color:#f39c12">[{timestamp}] 🛤️ Decoy routing: 3 real + {intensity} decoy packets</span>')
            elif step == 4:
                logs.append(f'<span style="color:#3498db">[{timestamp}] 🕸️ Gossip protocol: Alert sent to {nodes} peers</span>')
            elif step == 5:
                logs.append(f'<span style="color:#f39c12">[{timestamp}] 📢 Consensus vote initiated: {nodes} nodes voting</span>')

        # Phase 2: Defense responding (30-60%)
        elif step < 12:
            phase = "🟡 DEFENSE RESPONDING"
            threat = 0.7 - (step - 6) * 0.05
            pkts = random.randint(50, 300) * intensity

            if step == 6:
                votes = nodes - random.randint(0, 1)
                logs.append(f'<span style="color:#2ecc71">[{timestamp}] ✅ Consensus reached: {votes}/{nodes} nodes agree — THREAT CONFIRMED</span>')
            elif step == 7:
                logs.append(f'<span style="color:#e74c3c">[{timestamp}] 📦 Micro-containment: Port 8080 BLOCKED</span>')
            elif step == 8:
                logs.append(f'<span style="color:#e74c3c">[{timestamp}] 📦 Micro-containment: IP 10.0.0.X BLOCKED</span>')
            elif step == 9:
                logs.append(f'<span style="color:#9b59b6">[{timestamp}] 🎯 MTD: Ports mutated (8080→{random.randint(10000,60000)})</span>')
            elif step == 10:
                logs.append(f'<span style="color:#9b59b6">[{timestamp}] 🎯 MTD: {random.randint(2,5)} honeypots deployed</span>')
            elif step == 11:
                logs.append(f'<span style="color:#e74c3c">[{timestamp}] 📦 Escalation: Source IP QUARANTINED</span>')

        # Phase 3: Attack neutralized (60-100%)
        else:
            phase = "🟢 ATTACK NEUTRALIZED"
            threat = max(0.05, 0.4 - (step - 12) * 0.05)
            pkts = random.randint(5, 50)

            if step == 12:
                logs.append(f'<span style="color:#2ecc71">[{timestamp}] 🛡️ Attack traffic dropping — defenses holding</span>')
            elif step == 14:
                logs.append(f'<span style="color:#2ecc71">[{timestamp}] ✅ Attacker honeypot captured — fingerprint logged</span>')
            elif step == 16:
                logs.append(f'<span style="color:#2ecc71">[{timestamp}] ✅ All nodes report CLEAR status</span>')
            elif step == 18:
                logs.append(f'<span style="color:#2ecc71">[{timestamp}] ✅ Threat level normalized — monitoring continues</span>')
            elif step == 19:
                logs.append(f'<span style="color:#2ecc71">[{timestamp}] 🏆 ATTACK FULLY NEUTRALIZED — System secure</span>')

        threat_levels.append(threat)
        packet_counts.append(pkts)

        # Update status
        threat_metric.metric("Threat Level", f"{threat:.0%}")
        nodes_metric.metric("Active Nodes", f"{nodes}/{nodes}")
        packets_metric.metric("Malicious Packets", sum(packet_counts))
        actions_metric.metric("Defense Actions", min(step + 1, 12))
        phase_display.markdown(f"**Phase:** {phase}")

        # Update log
        log_container.markdown("<br>".join(logs[-8:]), unsafe_allow_html=True)

        # Update threat chart
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            y=threat_levels, mode="lines", fill="tozeroy",
            line=dict(color="#e74c3c", width=2), fillcolor="rgba(231,76,60,0.2)",
        ))
        fig1.add_hline(y=0.5, line_dash="dash", line_color="#f39c12", opacity=0.5)
        fig1.update_layout(
            title="Threat Level", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", font_color="#fafafa",
            yaxis=dict(range=[0, 1], tickformat=".0%"), xaxis_title="Time Step",
            height=250, margin=dict(t=40, b=20, l=40, r=20),
        )
        threat_chart.plotly_chart(fig1, use_container_width=True)

        # Update packet chart
        colors = ["#e74c3c" if p > 200 else "#f39c12" if p > 50 else "#2ecc71" for p in packet_counts]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(y=packet_counts, marker_color=colors))
        fig2.update_layout(
            title="Malicious Packets/s", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", font_color="#fafafa",
            xaxis_title="Time Step", height=250, margin=dict(t=40, b=20, l=40, r=20),
        )
        packet_chart.plotly_chart(fig2, use_container_width=True)

        time.sleep(delay)

    progress.progress(1.0)

    # ── Final Report ──
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("## 📋 Incident Report")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Total Duration", f"{total_steps * 500}ms")
    r2.metric("Detection Time", "500ms")
    r3.metric("Containment Time", f"{blocked_at * 500}ms")
    r4.metric("Result", "✅ NEUTRALIZED")

    st.markdown("### 🛡️ Defense Actions Taken")
    actions_taken = [
        ("🤖 ML Detection", "XGBoost classified attack with 99.58% confidence", "500ms"),
        ("🔐 Stealth Alert", "AES-256-GCM encrypted, HTTP camouflaged", "1000ms"),
        ("🛤️ Decoy Routing", f"Alert routed + {intensity} decoy packets injected", "1500ms"),
        ("🕸️ Mesh Consensus", f"{nodes}/{nodes} nodes voted to confirm threat", "3000ms"),
        ("📦 Containment", "Port block → IP block → Full quarantine", f"{blocked_at * 500}ms"),
        ("🎯 MTD Mutation", "All ports shuffled, honeypots deployed", f"{(blocked_at + 1) * 500}ms"),
    ]
    for icon_name, desc, timing in actions_taken:
        st.markdown(f"  **{icon_name}** — {desc} *(at {timing})*")

    st.success(
        f"🏆 **{attack_name}** neutralized in {total_steps * 500}ms with zero data loss. "
        f"All {nodes} nodes are secure."
    )
