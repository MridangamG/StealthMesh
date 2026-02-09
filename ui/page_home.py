"""
Home Page - StealthMesh Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os


def render():
    # Hero Section
    st.markdown('<h1 class="hero-title">🛡️ StealthMesh</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Adaptive Stealth Communication & Decentralized Defense for MSMEs</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Key Metrics Row ──
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Best Accuracy", "99.58%", "XGBoost")
    c2.metric("Datasets", "4", "902K samples")
    c3.metric("Models Trained", "12", "3 per dataset")
    c4.metric("Defense Modules", "6", "All active")
    c5.metric("Response Time", "<5 sec", "Automated")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Performance Overview Chart ──
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("📊 Model Accuracy Across Datasets")
        csv_path = os.path.join("results", "multi_dataset_comparison.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            fig = px.bar(
                df,
                x="Dataset",
                y="Accuracy",
                color="Model",
                barmode="group",
                color_discrete_sequence=["#e74c3c", "#3498db", "#2ecc71"],
                text="Accuracy",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#fafafa",
                yaxis=dict(range=[94, 101], title="Accuracy (%)"),
                xaxis_title="",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=400,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run `train_all_models.py` first to see results.")

    with col_right:
        st.subheader("📈 Dataset Coverage")
        dataset_info = {
            "Dataset": ["CICIDS 2017", "Network 10-Class", "Ransomware"],
            "Samples": [45365, 211043, 149043],
        }
        dfd = pd.DataFrame(dataset_info)
        fig2 = px.pie(
            dfd,
            values="Samples",
            names="Dataset",
            color_discrete_sequence=["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"],
            hole=0.45,
        )
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#fafafa",
            showlegend=False,
            height=400,
            margin=dict(t=30, b=20),
            annotations=[
                dict(text="902K<br>Total", x=0.5, y=0.5, font_size=16, showarrow=False, font_color="#fafafa")
            ],
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Defense Modules Grid ──
    st.subheader("🛡️ StealthMesh Defense Modules")
    modules = [
        ("🔐", "Stealth Communication", "Polymorphic AES-256 encryption with HTTP/DNS traffic camouflage"),
        ("🛤️", "Decoy Routing", "Dynamic multi-hop paths with fake traffic injection"),
        ("🕸️", "Mesh Coordinator", "Peer discovery, gossip protocol & consensus voting"),
        ("🤖", "Threat Detector", "ML-based real-time attack classification (<10ms)"),
        ("📦", "Micro-Containment", "Auto-escalation: warn → block → quarantine"),
        ("🎯", "Adaptive MTD", "Dynamic port mutation & honeypot deployment"),
    ]

    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(modules):
        with cols[i % 3]:
            st.markdown(
                f"""<div class="module-card">
                <h4>{icon} {title}</h4>
                <p>{desc}</p>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Quick Attack Response Flow ──
    st.subheader("⚡ Attack Response Pipeline")
    steps = [
        ("1️⃣", "Traffic Arrives", "Network packets enter the system"),
        ("2️⃣", "ML Detection", "XGBoost classifies with 99.58% accuracy"),
        ("3️⃣", "Stealth Alert", "Encrypted alert via camouflaged channel"),
        ("4️⃣", "Mesh Consensus", "Peer nodes vote on threat validity"),
        ("5️⃣", "Auto-Contain", "Offender IP/port blocked progressively"),
        ("6️⃣", "MTD Mutation", "Ports shuffled, honeypots deployed"),
    ]
    cols2 = st.columns(6)
    for i, (num, title, desc) in enumerate(steps):
        with cols2[i]:
            st.markdown(
                f"""<div class="module-card" style="text-align:center; min-height:140px;">
                <h3 style="margin:0; color:#e74c3c;">{num}</h3>
                <h4 style="margin:5px 0; color:#ffffff; font-size:0.9rem;">{title}</h4>
                <p style="font-size:0.75rem;">{desc}</p>
                </div>""",
                unsafe_allow_html=True,
            )
