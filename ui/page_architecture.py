"""
Architecture Page - StealthMesh Dashboard
"""

import streamlit as st
import plotly.graph_objects as go


def render():
    st.markdown('<h1 class="hero-title">🏗️ System Architecture</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Visual breakdown of StealthMesh\'s defense architecture</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🏗️ Architecture Diagram", "🔄 Attack Response Flow", "🗂️ Data Pipeline"])

    # ═══════════════════════════════════════════════
    # TAB 1: ARCHITECTURE
    # ═══════════════════════════════════════════════
    with tab1:
        st.subheader("🏗️ StealthMesh Layered Architecture")

        # Build layered architecture using Plotly
        fig = go.Figure()

        layers = [
            (0.0, 0.15, "NETWORK LAYER", "#2c3e50",
             "Packet Capture · Flow Analysis · Traffic Monitoring"),
            (0.18, 0.33, "DETECTION LAYER", "#e74c3c",
             "XGBoost (99.58%) · Random Forest · Neural Network · Feature Extraction"),
            (0.36, 0.51, "COMMUNICATION LAYER", "#3498db",
             "AES-256-GCM Encryption · HTTP/DNS Camouflage · Polymorphic Cipher Rotation"),
            (0.54, 0.69, "COORDINATION LAYER", "#2ecc71",
             "Mesh Network · Gossip Protocol · Consensus Voting · Trust Scoring"),
            (0.72, 0.87, "RESPONSE LAYER", "#9b59b6",
             "Micro-Containment · Port/IP Blocking · Quarantine · Auto-Escalation"),
            (0.90, 1.0, "DECEPTION LAYER", "#f39c12",
             "Adaptive MTD · Port Mutation · Honeypot Deployment · Surface Mutation"),
        ]

        for y0, y1, label, color, desc in layers:
            fig.add_shape(
                type="rect", x0=0.05, x1=0.95, y0=y0, y1=y1,
                fillcolor=color, opacity=0.3, line=dict(color=color, width=2),
            )
            fig.add_annotation(
                x=0.12, y=(y0 + y1) / 2, text=f"<b>{label}</b>",
                showarrow=False, font=dict(color=color, size=13),
                xanchor="left",
            )
            fig.add_annotation(
                x=0.95, y=(y0 + y1) / 2, text=desc,
                showarrow=False, font=dict(color="#aabbcc", size=10),
                xanchor="right",
            )

        # Arrows between layers
        for i in range(len(layers) - 1):
            y_from = layers[i][1]
            y_to = layers[i + 1][0]
            fig.add_annotation(
                x=0.5, y=(y_from + y_to) / 2, text="⬆",
                showarrow=False, font=dict(size=14, color="#555"),
            )

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False, range=[0, 1]),
            yaxis=dict(visible=False, range=[-0.05, 1.05]),
            height=700,
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Module relationships
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.subheader("🔗 Module Interactions")

        interactions = [
            ("Threat Detector", "→", "Mesh Coordinator", "Sends alert when attack detected"),
            ("Threat Detector", "→", "Micro-Containment", "Triggers containment for threat"),
            ("Threat Detector", "→", "Adaptive MTD", "Updates threat level"),
            ("Mesh Coordinator", "→", "All Peers", "Broadcasts alert via gossip"),
            ("Mesh Coordinator", "→", "Micro-Containment", "Preemptive blocking from peer alerts"),
            ("Micro-Containment", "→", "Mesh Coordinator", "Shares containment rules"),
            ("Stealth Comm", "→", "All Modules", "Encrypts inter-module communication"),
            ("Decoy Routing", "→", "Stealth Comm", "Routes messages via hidden paths"),
            ("Adaptive MTD", "→", "Mesh Coordinator", "Shares port changes with peers"),
        ]

        cols = st.columns([2, 1, 2, 4])
        cols[0].markdown("**Source**")
        cols[1].markdown("**→**")
        cols[2].markdown("**Target**")
        cols[3].markdown("**Purpose**")

        for src, arrow, tgt, purpose in interactions:
            cols = st.columns([2, 1, 2, 4])
            cols[0].markdown(f"`{src}`")
            cols[1].markdown(arrow)
            cols[2].markdown(f"`{tgt}`")
            cols[3].markdown(purpose)

    # ═══════════════════════════════════════════════
    # TAB 2: ATTACK RESPONSE FLOW
    # ═══════════════════════════════════════════════
    with tab2:
        st.subheader("🔄 Complete Attack Response Pipeline")

        steps = [
            ("1", "🌐 Traffic Arrives", "Network packets enter the monitored system",
             "All incoming/outgoing network flows are captured and converted to feature vectors (78 features from CICIDS format)"),
            ("2", "🤖 ML Detection", "XGBoost model analyzes traffic",
             "Trained model classifies traffic as Benign or Attack with 99.58% accuracy in <10ms. Outputs confidence score."),
            ("3", "🔐 Stealth Alert", "Alert encrypted and camouflaged",
             "Threat alert is encrypted using AES-256-GCM and wrapped in HTTP camouflage to look like normal web traffic."),
            ("4", "🛤️ Decoy Routing", "Alert sent via hidden path",
             "Message routed through random multi-hop path. Fake decoy packets injected to confuse any eavesdropper."),
            ("5", "🕸️ Mesh Consensus", "Peer nodes vote on threat",
             f"Alert propagated via gossip protocol. Nodes independently verify and vote. 67%+ agreement required."),
            ("6", "📦 Auto-Contain", "Progressive threat isolation",
             "1st offense: Warning → 2nd: Port block → 3rd: IP block → 4th: Full quarantine. All automatic."),
            ("7", "🎯 MTD Mutation", "Attack surface changes",
             "All service ports shuffled to new random values. Honeypot services deployed on old ports to trap attacker."),
            ("8", "✅ Neutralized", "System secured",
             "Attack blocked, attacker trapped in honeypot, all nodes updated. Total response time: <5 seconds."),
        ]

        for num, title, subtitle, detail in steps:
            with st.expander(f"**Step {num}: {title}** — {subtitle}", expanded=(num in ["1", "2"])):
                st.markdown(detail)

        # Timeline diagram
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.subheader("⏱️ Response Timeline")

        fig_timeline = go.Figure()
        times = [0, 500, 1000, 1500, 2500, 3500, 4500, 5000]
        labels_short = ["Detect", "Classify", "Encrypt", "Route", "Consensus", "Contain", "Mutate", "Secure"]
        colors = ["#e74c3c", "#e74c3c", "#3498db", "#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#2ecc71"]

        fig_timeline.add_trace(go.Scatter(
            x=times, y=[1]*8, mode="markers+text",
            marker=dict(size=20, color=colors, symbol="diamond"),
            text=labels_short, textposition="top center",
            textfont=dict(color="#fafafa", size=11),
        ))
        fig_timeline.add_trace(go.Scatter(
            x=times, y=[1]*8, mode="lines",
            line=dict(color="#3d4f6f", width=2, dash="dot"),
            showlegend=False,
        ))
        fig_timeline.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#fafafa",
            xaxis=dict(title="Time (milliseconds)", dtick=500),
            yaxis=dict(visible=False),
            showlegend=False,
            height=200,
            margin=dict(t=40, b=40, l=40, r=40),
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    # ═══════════════════════════════════════════════
    # TAB 3: DATA PIPELINE
    # ═══════════════════════════════════════════════
    with tab3:
        st.subheader("🗂️ ML Data Pipeline")

        pipeline_steps = [
            ("📥 Data Loading", [
                "Load CSV files (CICIDS 2017, Network, Ransomware)",
                "Merge multiple CSV files per dataset",
                "Total: 405,451 samples across 3 datasets",
            ]),
            ("🧹 Data Cleaning", [
                "Remove NaN values and infinite values",
                "Remove duplicate rows",
                "Handle missing data with median imputation",
                "Strip whitespace from column names",
            ]),
            ("🏷️ Label Encoding", [
                "Binary: BENIGN=0, Attack=1",
                "Multi-class: LabelEncoder for 10+ classes",
            ]),
            ("⚙️ Feature Engineering", [
                "Feature selection using mutual information",
                "Select top-K features (40 for CICIDS, varies per dataset)",
                "Remove low-variance features",
            ]),
            ("📏 Scaling", [
                "StandardScaler normalization",
                "Zero mean, unit variance",
                "Scaler saved for inference-time use",
            ]),
            ("✂️ Train/Test Split", [
                "80% training, 20% testing",
                "Stratified split to preserve class ratios",
                "Random seed for reproducibility",
            ]),
            ("💾 Save Artifacts", [
                "X_train.npy, X_test.npy, y_train.npy, y_test.npy",
                "Scaler pickle, feature names pickle",
                "Label mapping pickle (for multi-class)",
            ]),
        ]

        for step_name, details in pipeline_steps:
            with st.expander(f"**{step_name}**"):
                for d in details:
                    st.markdown(f"  • {d}")

        # Dataset summary table
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.subheader("📊 Dataset Summary")

        import pandas as pd
        ds = pd.DataFrame({
            "Dataset": ["CICIDS 2017", "Network 10-Class", "Ransomware"],
            "Samples": ["45,365", "211,043", "149,043"],
            "Features": [40, 27, 7],
            "Classes": [2, 10, 3],
            "Type": ["Binary", "Multi-class", "Multi-class"],
            "Best Model": ["XGBoost", "RandomForest", "RandomForest"],
            "Best Accuracy": ["99.58%", "98.94%", "97.73%"],
        })
        st.dataframe(ds, use_container_width=True, hide_index=True)
