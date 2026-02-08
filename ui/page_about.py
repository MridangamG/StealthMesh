"""
About Page - StealthMesh Dashboard
"""

import streamlit as st


def render():
    st.markdown('<h1 class="hero-title">📄 About StealthMesh</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Project information and research contributions</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Project Overview ──
    st.subheader("🎯 Project Overview")
    st.markdown("""
    **StealthMesh** is a stealth-enabled, decentralized cyber defense framework tailored for
    Micro, Small, and Medium Enterprises (MSMEs). The system integrates advanced stealth
    communication techniques with a decentralized mesh defense protocol to provide affordable,
    lightweight, and adaptive protection against sophisticated cyber threats.
    """)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Research Novelty ──
    st.subheader("🔬 Research Novelty (What Makes This Unique)")

    novelty = [
        ("🔐 Stealth-Enabled Defense",
         "No other paper combines ML-based detection with stealth defense communication. "
         "Our polymorphic encryption and traffic camouflage make the defense itself invisible to attackers.",
         "Other Papers: Defense systems are visible and can be detected/bypassed"),
        ("🕸️ Decentralized Mesh Architecture",
         "Unlike centralized IDS/SIEM systems that have a single point of failure, "
         "StealthMesh uses a peer-to-peer mesh topology. Even if 40% of nodes fail, defense continues.",
         "Other Papers: Rely on centralized server architecture"),
        ("⚡ Autonomous Response",
         "Most research papers stop at detection ('We achieved 99% accuracy'). "
         "StealthMesh closes the gap between detection and action with sub-second automated response.",
         "Other Papers: No response mechanism; require human intervention"),
        ("🎯 Moving Target Defense + ML",
         "First framework to integrate ML threat detection with dynamic port mutation "
         "and honeypot deployment. Attacker reconnaissance becomes useless.",
         "Other Papers: Static system configuration"),
        ("📊 Multi-Dataset Validation",
         "Validated across 4 diverse datasets (902,451 samples) covering DDoS, ransomware, "
         "zero-day, and multi-class attacks. Proves generalization beyond a single dataset.",
         "Other Papers: Typically test on only 1 dataset"),
        ("💰 MSME-Focused Design",
         "Designed for resource-constrained environments. Lightweight, no specialized hardware, "
         "zero expertise required. Addresses the $50B MSME cybersecurity market gap.",
         "Other Papers: Enterprise-grade solutions costing $50K-$500K/year"),
    ]

    for title, desc, comparison in novelty:
        with st.expander(f"**{title}**", expanded=False):
            st.markdown(desc)
            st.markdown(f"*{comparison}*")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Competitive Comparison ──
    st.subheader("📊 Competitive Comparison")

    import pandas as pd
    comp = pd.DataFrame({
        "Feature": ["ML Detection", "Auto Response", "Stealth Mode", "Decentralized",
                    "Moving Target Defense", "MSME Affordable", "Zero Expertise"],
        "Snort/Suricata": ["✅ Rules-based", "❌ Manual", "❌", "❌", "❌", "✅ Free", "❌"],
        "Enterprise SIEM": ["✅ ML-based", "⚠️ Partial", "❌", "❌", "❌", "❌ $50K+", "❌"],
        "Cloud Security": ["✅ ML-based", "⚠️ Partial", "❌", "❌", "❌", "⚠️ $500/mo", "⚠️"],
        "StealthMesh (Ours)": ["✅ ML 99.58%", "✅ Full Auto", "✅ Yes", "✅ Yes", "✅ Yes", "✅ ~$50/mo", "✅ Yes"],
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Technology Stack ──
    st.subheader("🛠️ Technology Stack")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        **Languages & Frameworks**
        - Python 3.8+
        - Streamlit (Dashboard)
        - Plotly (Visualizations)
        """)
    with c2:
        st.markdown("""
        **ML Libraries**
        - scikit-learn
        - XGBoost
        - NumPy / Pandas
        """)
    with c3:
        st.markdown("""
        **Security**
        - AES-256-GCM/CBC
        - HMAC-SHA256
        - Cryptography library
        """)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Key Phrases ──
    st.subheader("📝 Key Phrases for Research Paper")
    phrases = [
        "To the best of our knowledge, this is the first work to integrate stealth communication with ML-based threat detection.",
        "Unlike existing approaches that rely on centralized architectures, StealthMesh employs a decentralized mesh topology.",
        "We bridge the gap between detection and response through autonomous micro-containment.",
        "Validated across four diverse datasets comprising 902,451 samples.",
        "Designed specifically for resource-constrained MSME environments.",
    ]
    for p in phrases:
        st.markdown(f"> *{p}*")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Future Roadmap ──
    st.subheader("🚀 Future Roadmap")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown("""
        <div class="module-card">
        <h4 style="color:#2ecc71">Phase 1 ✅</h4>
        <p><b>Research</b><br>ML models trained<br>4 datasets validated<br>Core modules built</p>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown("""
        <div class="module-card">
        <h4 style="color:#3498db">Phase 2</h4>
        <p><b>Prototype</b><br>Real network deploy<br>Performance tuning<br>UI development</p>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown("""
        <div class="module-card">
        <h4 style="color:#f39c12">Phase 3</h4>
        <p><b>Pilot</b><br>10 MSME deployment<br>Feedback collection<br>Iterate & improve</p>
        </div>
        """, unsafe_allow_html=True)
    with r4:
        st.markdown("""
        <div class="module-card">
        <h4 style="color:#e74c3c">Phase 4</h4>
        <p><b>Launch</b><br>SaaS product<br>Mobile alerts<br>24/7 monitoring</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center; color:#556;">StealthMesh © 2026 — '
        'Adaptive Stealth Communication and Decentralized Defense for MSMEs</p>',
        unsafe_allow_html=True,
    )
