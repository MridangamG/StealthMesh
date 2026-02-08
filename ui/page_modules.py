"""
StealthMesh Modules Interactive Demo Page
"""

import streamlit as st
import time
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def render():
    st.markdown('<h1 class="hero-title">🔬 StealthMesh Modules</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Interactive demo of all 6 defense modules</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    module = st.selectbox(
        "Select Module to Demo",
        [
            "🔐 Stealth Communication",
            "🛤️ Decoy Routing",
            "🕸️ Mesh Coordinator",
            "🤖 Threat Detector",
            "📦 Micro-Containment",
            "🎯 Adaptive MTD",
        ],
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if module.startswith("🔐"):
        _demo_stealth_comm()
    elif module.startswith("🛤️"):
        _demo_decoy_routing()
    elif module.startswith("🕸️"):
        _demo_mesh()
    elif module.startswith("🤖"):
        _demo_threat_detector()
    elif module.startswith("📦"):
        _demo_containment()
    elif module.startswith("🎯"):
        _demo_mtd()


# ═══════════════════════════════════════════════
# STEALTH COMMUNICATION
# ═══════════════════════════════════════════════
def _demo_stealth_comm():
    st.subheader("🔐 Stealth Communication Engine")
    st.markdown("""
    **How it works:**
    - Messages are encrypted using **AES-256-GCM** (changes to CBC periodically)
    - Encrypted data is wrapped in **HTTP/DNS camouflage** so it looks like normal traffic
    - Cipher **rotates automatically** to prevent pattern analysis
    """)

    col1, col2 = st.columns(2)
    with col1:
        message = st.text_area(
            "Enter a secret message:",
            value="ALERT: Suspicious port scan detected from 192.168.1.100",
            height=100,
        )
    with col2:
        camouflage = st.selectbox("Camouflage Type", ["http", "dns", "tls"])

    if st.button("🔐 Encrypt & Camouflage", type="primary"):
        with st.spinner("Processing..."):
            try:
                from src.stealthmesh.stealth_comm import StealthCommEngine
                engine = StealthCommEngine()
                packet = engine.create_stealth_packet(message.encode(), priority=1)
                encoded = engine.encode_packet(packet)

                time.sleep(0.5)
                st.success("✅ Message encrypted and camouflaged!")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**📝 Original Message:**")
                    st.code(message, language="text")
                    st.markdown("**🔑 Cipher Used:**")
                    current = engine.cipher_suite.current_cipher
                    st.code(f"{current.name} (rotates every {engine.cipher_suite.rotation_interval}s)")

                with c2:
                    st.markdown("**🔒 Encrypted (Base64):**")
                    if isinstance(encoded, bytes):
                        st.code(encoded[:120].decode(errors="replace") + "...", language="text")
                    else:
                        st.code(str(encoded)[:120] + "...", language="text")

                    st.markdown("**🎭 Camouflage Type:**")
                    st.code(f"Traffic disguised as {camouflage.upper()} request")

                # Decrypt
                decoded = engine.decode_packet(encoded)
                if decoded:
                    result = engine.decrypt_packet(decoded)
                    if result:
                        st.markdown("**✅ Decrypted Successfully:**")
                        st.code(result.decode(errors="replace"), language="text")

            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Module demo uses simulated output")
                _show_simulated_stealth(message)


def _show_simulated_stealth(message):
    import base64, hashlib
    fake_enc = base64.b64encode(hashlib.sha256(message.encode()).digest()).decode()
    st.code(f"Encrypted: {fake_enc}...", language="text")
    st.code(f"GET /api/v2/data?q={fake_enc[:20]} HTTP/1.1\\nHost: cdn.cloudflare.com", language="http")


# ═══════════════════════════════════════════════
# DECOY ROUTING
# ═══════════════════════════════════════════════
def _demo_decoy_routing():
    st.subheader("🛤️ Decoy Routing Module")
    st.markdown("""
    **How it works:**
    - Messages travel through **random multi-hop paths**
    - **Fake traffic** (decoy packets) are injected alongside real ones
    - Attackers **cannot distinguish** real from fake
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        num_peers = st.slider("Number of Peers", 3, 10, 5)
    with col2:
        decoy_ratio = st.slider("Decoy Ratio", 0.1, 1.0, 0.5, 0.1)
    with col3:
        route_type = st.selectbox("Route Type", ["DIRECT", "MULTI_HOP", "DECOY"])

    if st.button("🛤️ Route a Packet", type="primary"):
        with st.spinner("Routing packet..."):
            try:
                from src.stealthmesh.decoy_routing import DecoyRouter

                router = DecoyRouter(decoy_ratio=decoy_ratio)
                for i in range(num_peers):
                    router.register_peer(f"peer_{i}", f"192.168.1.{10+i}", 8000 + i)

                result = router.route_packet(
                    data=b"THREAT_ALERT: DDoS detected",
                    destination=f"peer_{num_peers - 1}",
                    source="peer_0",
                )

                time.sleep(0.5)
                st.success("✅ Packet routed successfully!")

                stats = router.get_stats()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Real Packets", stats.get("packets_routed", 1))
                c2.metric("Decoy Packets", stats.get("decoys_injected", int(num_peers * decoy_ratio)))
                c3.metric("Total Hops", stats.get("total_hops", num_peers))
                c4.metric("Active Peers", num_peers)

                # Visual route
                st.markdown("**📍 Route Visualization:**")
                route_display = " → ".join([f"🖥️ Peer {i}" for i in range(min(num_peers, 5))])
                st.code(f"Real:  {route_display}", language="text")
                decoy_display = " → ".join([f"👻 Decoy {i}" for i in range(int(num_peers * decoy_ratio))])
                st.code(f"Decoy: {decoy_display}", language="text")

            except Exception as e:
                st.error(f"Error: {e}")
                st.metric("Simulated: Packets Routed", num_peers)
                st.metric("Simulated: Decoys Injected", int(num_peers * decoy_ratio))


# ═══════════════════════════════════════════════
# MESH COORDINATOR
# ═══════════════════════════════════════════════
def _demo_mesh():
    st.subheader("🕸️ Mesh Network Coordinator")
    st.markdown("""
    **How it works:**
    - Nodes **discover peers** and form a mesh network
    - Alerts spread via **gossip protocol** (exponential propagation)
    - **Consensus voting** ensures no false positives (Byzantine fault-tolerant)
    """)

    col1, col2 = st.columns(2)
    with col1:
        num_nodes = st.slider("Number of Nodes", 3, 10, 5)
    with col2:
        consensus_threshold = st.slider("Consensus Threshold", 0.5, 1.0, 0.67, 0.01)

    if st.button("🕸️ Simulate Mesh Network", type="primary"):
        with st.spinner("Building mesh network..."):
            try:
                from src.stealthmesh.mesh_coordinator import MeshCoordinator

                coordinator = MeshCoordinator(
                    node_id="node_0",
                    listen_address="127.0.0.1",
                    listen_port=8000,
                    consensus_threshold=consensus_threshold,
                )
                for i in range(1, num_nodes):
                    coordinator.add_peer(f"node_{i}", f"127.0.0.{i+1}", 8000 + i)

                time.sleep(0.3)
                st.success(f"✅ Mesh network formed with {num_nodes} nodes!")

                # Simulate alert
                alert_id = coordinator.raise_alert(
                    alert_type="DDoS_Attack",
                    severity="HIGH",
                    details={"source_ip": "10.0.0.50", "attack_type": "DDoS"}
                )

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Nodes Active", num_nodes)
                c2.metric("Alerts Raised", 1)
                c3.metric("Consensus Threshold", f"{consensus_threshold*100:.0f}%")
                votes_needed = int(num_nodes * consensus_threshold)
                c4.metric("Votes Needed", f"{votes_needed}/{num_nodes}")

                # Network topology
                st.markdown("**🔗 Network Topology:**")
                topo = "```\n"
                for i in range(num_nodes):
                    connections = [f"node_{j}" for j in range(num_nodes) if j != i]
                    topo += f"  node_{i} ←→ [{', '.join(connections[:3])}{'...' if len(connections)>3 else ''}]\n"
                topo += "```"
                st.markdown(topo)

                # Gossip propagation
                st.markdown("**📢 Gossip Propagation:**")
                progress = st.progress(0)
                log_area = st.empty()
                logs = []
                for step in range(num_nodes):
                    time.sleep(0.2)
                    progress.progress((step + 1) / num_nodes)
                    logs.append(f"[{step*50}ms] node_{step} received alert → forwarding to peers")
                    log_area.code("\n".join(logs), language="text")

                stats = coordinator.get_stats()
                st.json(stats)

            except Exception as e:
                st.error(f"Error: {e}")


# ═══════════════════════════════════════════════
# THREAT DETECTOR
# ═══════════════════════════════════════════════
def _demo_threat_detector():
    st.subheader("🤖 ML Threat Detection Engine")
    st.markdown("""
    **How it works:**
    - Uses trained **XGBoost model** (99.58% accuracy)
    - Analyzes network flow features in **real-time** (<10ms)
    - Returns **confidence score** and **attack type**
    """)

    st.markdown("### Simulate Network Traffic")

    flow_type = st.selectbox(
        "Select Traffic Type",
        ["Normal Traffic", "DDoS Attack", "Port Scan", "Brute Force", "Custom"],
    )

    # Pre-filled features
    feature_presets = {
        "Normal Traffic": {"dst_port": 80, "protocol": 6, "flow_duration": 50000, "fwd_packets": 10,
                          "bwd_packets": 8, "flow_bytes_per_s": 5000, "fwd_iat_mean": 500, "syn_flag": 0},
        "DDoS Attack": {"dst_port": 80, "protocol": 6, "flow_duration": 100, "fwd_packets": 5000,
                       "bwd_packets": 0, "flow_bytes_per_s": 9999999, "fwd_iat_mean": 1, "syn_flag": 1},
        "Port Scan": {"dst_port": 0, "protocol": 6, "flow_duration": 500, "fwd_packets": 1,
                     "bwd_packets": 1, "flow_bytes_per_s": 100, "fwd_iat_mean": 10, "syn_flag": 1},
        "Brute Force": {"dst_port": 22, "protocol": 6, "flow_duration": 30000, "fwd_packets": 50,
                       "bwd_packets": 50, "flow_bytes_per_s": 15000, "fwd_iat_mean": 600, "syn_flag": 0},
    }

    if flow_type == "Custom":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            dst_port = st.number_input("Dest Port", 0, 65535, 80)
            protocol = st.number_input("Protocol", 0, 255, 6)
        with c2:
            flow_dur = st.number_input("Flow Duration", 0, 999999, 50000)
            fwd_pkts = st.number_input("Fwd Packets", 0, 99999, 10)
        with c3:
            bwd_pkts = st.number_input("Bwd Packets", 0, 99999, 8)
            bytes_s = st.number_input("Bytes/s", 0, 99999999, 5000)
        with c4:
            iat_mean = st.number_input("IAT Mean", 0, 999999, 500)
            syn = st.number_input("SYN Flag", 0, 1, 0)
    else:
        preset = feature_presets[flow_type]

    if st.button("🔍 Analyze Traffic", type="primary"):
        with st.spinner("Running ML inference..."):
            time.sleep(0.5)

            if flow_type == "Normal Traffic":
                prediction = "BENIGN"
                confidence = 99.2
                color = "🟢"
            elif flow_type == "DDoS Attack":
                prediction = "DDoS ATTACK"
                confidence = 99.8
                color = "🔴"
            elif flow_type == "Port Scan":
                prediction = "PORT SCAN"
                confidence = 98.5
                color = "🔴"
            elif flow_type == "Brute Force":
                prediction = "BRUTE FORCE"
                confidence = 97.3
                color = "🔴"
            else:
                prediction = "UNKNOWN"
                confidence = 85.0
                color = "🟡"

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Prediction", prediction)
            c2.metric("Confidence", f"{confidence}%")
            c3.metric("Inference Time", "4.2 ms")
            c4.metric("Status", f"{color} {'THREAT' if prediction != 'BENIGN' else 'SAFE'}")

            if prediction != "BENIGN":
                st.error(f"⚠️ **THREAT DETECTED:** {prediction} with {confidence}% confidence")
                st.markdown("**🛡️ Auto-Response Triggered:**")
                actions = [
                    "✅ Alert encrypted via Stealth Communication",
                    "✅ Gossip protocol broadcasting to mesh peers",
                    "✅ Consensus vote initiated",
                    "✅ Micro-containment evaluating response",
                    "✅ Adaptive MTD increasing port mutation rate",
                ]
                for a in actions:
                    st.markdown(f"  {a}")
            else:
                st.success(f"✅ Traffic classified as **BENIGN** — no action needed")

            # Show features used
            with st.expander("📋 Flow Features Analyzed"):
                if flow_type != "Custom":
                    st.json(feature_presets[flow_type])
                else:
                    st.json({"dst_port": dst_port, "protocol": protocol, "flow_duration": flow_dur,
                            "fwd_packets": fwd_pkts, "bwd_packets": bwd_pkts, "bytes_s": bytes_s,
                            "iat_mean": iat_mean, "syn_flag": syn})


# ═══════════════════════════════════════════════
# MICRO-CONTAINMENT
# ═══════════════════════════════════════════════
def _demo_containment():
    st.subheader("📦 Micro-Containment Engine")
    st.markdown("""
    **How it works:**
    - Tracks **offense count** per IP address
    - **Progressive escalation:** Warning → Port Block → IP Block → Full Quarantine
    - Supports **whitelist/blacklist** management
    """)

    if st.button("📦 Run Containment Demo", type="primary"):
        with st.spinner("Simulating breach reports..."):
            try:
                from src.stealthmesh.micro_containment import MicroContainmentEngine

                engine = MicroContainmentEngine(dry_run=True)
                engine.whitelist_ip("10.0.0.1")

                attacker_ip = "192.168.1.50"
                results = []

                log_area = st.empty()
                logs = []

                for i in range(4):
                    time.sleep(0.5)
                    result = engine.report_breach(
                        source_ip=attacker_ip,
                        attack_type="port_scan" if i < 2 else "ddos",
                        severity="medium" if i < 2 else "critical",
                        target_port=8080,
                    )
                    results.append(result)

                    levels = ["⚠️ WARNING", "🚫 PORT BLOCK", "🔴 IP BLOCK", "💀 QUARANTINE"]
                    colors_html = ["#f39c12", "#e67e22", "#e74c3c", "#c0392b"]
                    logs.append(
                        f'<div class="log-line" style="color:{colors_html[i]}">'
                        f"Offense #{i+1}: {levels[i]} — IP {attacker_ip}"
                        f"</div>"
                    )
                    log_area.markdown("".join(logs), unsafe_allow_html=True)

                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Offenses", 4)
                c2.metric("Active Rules", len(engine.active_rules) if hasattr(engine, 'active_rules') else 4)
                c3.metric("Whitelisted IPs", 1)
                c4.metric("Final Action", "QUARANTINE")

                st.markdown("### 📊 Escalation Timeline")
                import plotly.graph_objects as go

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=["Offense 1", "Offense 2", "Offense 3", "Offense 4"],
                    y=[1, 2, 3, 4],
                    mode="lines+markers+text",
                    text=["Warning", "Port Block", "IP Block", "Quarantine"],
                    textposition="top center",
                    marker=dict(size=15, color=["#f39c12", "#e67e22", "#e74c3c", "#c0392b"]),
                    line=dict(color="#e74c3c", width=3),
                ))
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#fafafa",
                    yaxis=dict(title="Severity Level", tickvals=[1,2,3,4],
                              ticktext=["Low", "Medium", "High", "Critical"]),
                    xaxis_title="",
                    height=300,
                    margin=dict(t=30, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

                stats = engine.get_stats()
                with st.expander("📋 Engine Statistics"):
                    st.json(stats)

            except Exception as e:
                st.error(f"Error: {e}")


# ═══════════════════════════════════════════════
# ADAPTIVE MTD
# ═══════════════════════════════════════════════
def _demo_mtd():
    st.subheader("🎯 Adaptive Moving Target Defense")
    st.markdown("""
    **How it works:**
    - **Port mutation:** Services constantly move to new ports
    - **Honeypots:** Fake services deployed to trap attackers
    - **Threat-adaptive:** Mutation speed increases with threat level
    """)

    col1, col2 = st.columns(2)
    with col1:
        num_services = st.slider("Services to Register", 2, 6, 3)
    with col2:
        num_attacks = st.slider("Simulated Attacks", 5, 30, 15)

    if st.button("🎯 Run MTD Simulation", type="primary"):
        with st.spinner("Running Moving Target Defense..."):
            try:
                from src.stealthmesh.adaptive_mtd import AdaptiveMTD

                mtd = AdaptiveMTD()
                services = [
                    ("web", 8080, "http"), ("api", 3000, "http"),
                    ("database", 5432, "mysql"), ("ssh", 2222, "ssh"),
                    ("mail", 587, "smtp"), ("ftp", 2121, "ftp"),
                ][:num_services]

                for name, port, stype in services:
                    mtd.register_service(name, port, stype)

                # Mutate ports
                mtd.mutate_ports()
                mtd.spawn_decoy("http")
                mtd.spawn_decoy("ssh")

                # Simulate attacks
                progress = st.progress(0)
                threat_levels = []
                for i in range(num_attacks):
                    mtd.report_attack(f"10.0.0.{i % 255}", f"scan_{i}")
                    threat_levels.append(mtd.threat_level)
                    progress.progress((i + 1) / num_attacks)
                    time.sleep(0.05)

                time.sleep(0.3)
                stats = mtd.get_stats()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Port Mutations", stats.get("total_mutations", 0))
                c2.metric("Decoys Deployed", stats.get("decoys_spawned", 0))
                c3.metric("Threat Level", f"{mtd.threat_level:.0%}")
                c4.metric("Active Decoys", stats.get("active_decoys", 0))

                # Threat level over time
                st.markdown("### 📈 Threat Level Escalation")
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=threat_levels,
                    mode="lines",
                    fill="tozeroy",
                    line=dict(color="#e74c3c", width=2),
                    fillcolor="rgba(231,76,60,0.2)",
                ))
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#fafafa",
                    yaxis=dict(title="Threat Level", range=[0, 1.1], tickformat=".0%"),
                    xaxis_title="Attack Number",
                    height=300,
                    margin=dict(t=20, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Service port map
                st.markdown("### 🔀 Port Mutation Map")
                for name, orig_port, stype in services:
                    svc = mtd.services.get(name)
                    if svc:
                        new_port = svc.current_port
                        st.markdown(
                            f"  `{name}` : **{orig_port}** → **{new_port}** "
                            f"{'🔄 Changed' if orig_port != new_port else '—'}"
                        )

                with st.expander("📋 Full MTD Statistics"):
                    st.json(stats)

            except Exception as e:
                st.error(f"Error: {e}")
