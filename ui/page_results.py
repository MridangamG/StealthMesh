"""
ML Model Results Page - StealthMesh Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import pickle
from sklearn.metrics import confusion_matrix


def render():
    st.markdown('<h1 class="hero-title">📊 ML Model Results</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Performance analysis across 4 datasets × 3 models</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    csv_path = os.path.join("results", "multi_dataset_comparison.csv")
    if not os.path.exists(csv_path):
        st.error("Results file not found. Run `train_all_models.py` first.")
        return

    df = pd.read_csv(csv_path)

    # ── Dataset selector ──
    tab_all, tab_single = st.tabs(["📊 All Datasets Comparison", "🔍 Single Dataset Deep-Dive"])

    # ═══════════════════════════════════════════════
    # TAB 1: ALL DATASETS
    # ═══════════════════════════════════════════════
    with tab_all:
        # Metrics summary row
        best = df.loc[df["Accuracy"].idxmax()]
        avg_acc = df["Accuracy"].mean()
        avg_f1 = df["F1-Score"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Accuracy", f"{best['Accuracy']:.2f}%", f"{best['Model']} on {best['Dataset']}")
        c2.metric("Average Accuracy", f"{avg_acc:.2f}%")
        c3.metric("Average F1-Score", f"{avg_f1:.2f}%")
        c4.metric("Total Models", "12", "3 models × 4 datasets")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Accuracy heatmap
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Accuracy Heatmap")
            pivot = df.pivot(index="Dataset", columns="Model", values="Accuracy")
            fig_heat = px.imshow(
                pivot,
                text_auto=".2f",
                color_continuous_scale=["#1a1f2e", "#e74c3c", "#2ecc71"],
                aspect="auto",
            )
            fig_heat.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#fafafa",
                height=350,
                margin=dict(t=30, b=20),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        with col2:
            st.subheader("⏱️ Training Time Comparison")
            fig_time = px.bar(
                df,
                x="Dataset",
                y="Training (s)",
                color="Model",
                barmode="group",
                color_discrete_sequence=["#e74c3c", "#3498db", "#2ecc71"],
                log_y=True,
            )
            fig_time.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#fafafa",
                yaxis_title="Time (seconds, log scale)",
                xaxis_title="",
                height=350,
                margin=dict(t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_time, use_container_width=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Radar chart per model
        st.subheader("🕸️ Model Performance Radar")
        models = df["Model"].unique()
        rcols = st.columns(len(models))
        metrics_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

        for i, model in enumerate(models):
            with rcols[i]:
                model_df = df[df["Model"] == model]
                avg_vals = [model_df[m].mean() for m in metrics_cols]
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=avg_vals + [avg_vals[0]],
                    theta=metrics_cols + [metrics_cols[0]],
                    fill="toself",
                    name=model,
                    line_color="#e74c3c" if model == "XGBoost" else "#3498db" if model == "RandomForest" else "#2ecc71",
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[94, 101], tickfont=dict(color="#888")),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#fafafa",
                    title=dict(text=model, font=dict(size=14)),
                    showlegend=False,
                    height=300,
                    margin=dict(t=50, b=20, l=40, r=40),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Full results table
        st.subheader("📋 Complete Results Table")
        styled = df.style.format({
            "Accuracy": "{:.2f}%",
            "Precision": "{:.2f}%",
            "Recall": "{:.2f}%",
            "F1-Score": "{:.2f}%",
            "ROC-AUC": "{:.2f}%",
            "Training (s)": "{:.2f}s",
        }).highlight_max(subset=["Accuracy", "F1-Score", "ROC-AUC"], color="#2d6b3f")
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════
    # TAB 2: SINGLE DATASET DEEP-DIVE
    # ═══════════════════════════════════════════════
    with tab_single:
        dataset_names = df["Dataset"].unique().tolist()
        selected = st.selectbox("Select Dataset", dataset_names)

        sub_df = df[df["Dataset"] == selected]
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Model comparison for this dataset
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"📊 {selected} — Metrics Comparison")
            metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
            fig_comp = go.Figure()
            colors = {"RandomForest": "#3498db", "XGBoost": "#e74c3c", "NeuralNetwork": "#2ecc71"}
            for _, row in sub_df.iterrows():
                fig_comp.add_trace(go.Bar(
                    name=row["Model"],
                    x=metrics,
                    y=[row[m] for m in metrics],
                    marker_color=colors.get(row["Model"], "#888"),
                    text=[f"{row[m]:.1f}" for m in metrics],
                    textposition="outside",
                ))
            fig_comp.update_layout(
                barmode="group",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#fafafa",
                yaxis=dict(range=[90, 101], title="Score (%)"),
                height=400,
                margin=dict(t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        with c2:
            st.subheader(f"⏱️ {selected} — Training Time")
            fig_t = px.bar(
                sub_df,
                x="Model",
                y="Training (s)",
                color="Model",
                color_discrete_map=colors,
                text="Training (s)",
            )
            fig_t.update_traces(texttemplate="%{text:.2f}s", textposition="outside")
            fig_t.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#fafafa",
                showlegend=False,
                yaxis_title="Time (seconds)",
                height=400,
                margin=dict(t=30, b=20),
            )
            st.plotly_chart(fig_t, use_container_width=True)

        # Confusion matrix
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.subheader(f"🔢 {selected} — Confusion Matrices")

        prefix_map = {
            "CICIDS_2017": "binary",
            "Network_10Class": "network_multiclass",
            "Ransomware": "ransomware",
        }
        model_map = {
            "RandomForest": "randomforest",
            "XGBoost": "xgboost",
            "NeuralNetwork": "neuralnetwork",
        }
        prefix = prefix_map.get(selected, "binary")

        # Load test data
        y_test_path = os.path.join("processed_data", f"{prefix}_y_test.npy")
        X_test_path = os.path.join("processed_data", f"{prefix}_X_test.npy")

        if os.path.exists(y_test_path) and os.path.exists(X_test_path):
            y_test = np.load(y_test_path)
            X_test = np.load(X_test_path)

            cm_cols = st.columns(3)
            for i, model_name in enumerate(["RandomForest", "XGBoost", "NeuralNetwork"]):
                model_file = os.path.join("models", f"{prefix}_{model_map[model_name]}_model.pkl")
                with cm_cols[i]:
                    if os.path.exists(model_file):
                        with open(model_file, "rb") as f:
                            model = pickle.load(f)
                        y_pred = model.predict(X_test)
                        cm = confusion_matrix(y_test, y_pred)
                        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

                        fig_cm = px.imshow(
                            cm_norm,
                            text_auto=".1f",
                            color_continuous_scale=["#0e1117", "#e74c3c"],
                            aspect="auto",
                        )
                        fig_cm.update_layout(
                            title=dict(text=model_name, font=dict(size=13)),
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            font_color="#fafafa",
                            xaxis_title="Predicted",
                            yaxis_title="Actual",
                            height=300,
                            margin=dict(t=40, b=20, l=20, r=20),
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)
                    else:
                        st.info(f"{model_name} model not found")
        else:
            st.info("Processed test data not found. Run preprocessing first.")
