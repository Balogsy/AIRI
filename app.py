import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shap  
import threading

np.random.seed(42)
lock = threading.Lock()

st.set_page_config(
    page_title="AIRI Governance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_ml_pipeline_and_explainer():
    """
    Safely load pre-trained Random Forest and feature configuration artifacts
    and pre-initialize the SHAP explainer within a single cached scope.
    """
    try:
        with open("rf_airi_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("feature_columns.pkl", "rb") as f:
            feature_cols = pickle.load(f)
        
        explainer = shap.TreeExplainer(model)
        return model, feature_cols, explainer, True
    except (FileNotFoundError, Exception):
        return None, None, None, False

rf_model, feature_cols, explainer, model_loaded = load_ml_pipeline_and_explainer()

INDICATOR_MAP = {
    "D1_Data_Quality": "IND-D1-01 Data Quality Monitoring",
    "D1_Data_Governance": "IND-D1-02 Data Lineage & Governance",
    "D1_Data_Integration": "IND-D1-03 System Integration Capacity",
    "D2_System_Capability": "IND-D2-01 Machine Learning Deployment Capability",
    "D2_AI_Tooling": "IND-D2-02 MLOps & Drift Monitoring",
    "D2_Infrastructure_Resilience": "IND-D2-03 Operational Resilience & Incident Handling",
    "D3_FCA_Alignment": "IND-D3-01 Documented FCA AI Alignment",
    "D3_Consumer_Duty": "IND-D3-02 Consumer Duty Outcomes Tracking",
    "D3_Audit_Trail": "IND-D3-03 Immutable Audit Logging",
    "D4_Talent_Readiness": "IND-D4-01 Role Availability & AI Literacy",
    "D4_Change_Management": "IND-D4-02 Structured Tech Change Control",
    "D4_Leadership_Commitment": "IND-D4-03 Executive Sponsorship & Budgeting",
    "D5_Bias_Mitigation": "IND-D5-01 Algorithmic Bias Detection Frameworks",
    "D5_Explainability": "IND-D5-02 Model Explainability Protocols",
    "D5_Accountability": "IND-D5-03 Escalation & Governance Committee Structure"
}

st.title("Artificial Intelligence Readiness Index (AIRI)")
st.caption(
    "An Information Systems Diagnostic & Governance Tool for UK Debt Management Operations"
)

st.info("""
This prototype AIRI dashboard was developed as a research-oriented decision-support artefact 
for MSc Information Technology dissertation purposes. 
The machine learning outputs are exploratory and based on synthetic institutional readiness data.
""")

tab_assessment, tab_performance, tab_empirical, tab_feedback = st.tabs([
    "📊 Institutional Assessment Tool",
    "⚙️ ML Engine Evaluation Metrics",
    "📈 Stage 1 Pre-Deployment Empirical Validation Results",
    "🧪 Stage 5 Expert Feedback (SUS + Thematic)"
])

with tab_assessment:
    st.markdown("### 🛠️ Real-time Assessment Simulator")
    st.write("""
    Users can adjust institutional scoring indicators across the five AIRI dimensions via the collapsible sidebar, enabling real-time recalculation of composite scores and machine learning predictions.
    """)
    
    st.sidebar.header("🎯 Framework Inputs (1.00 - 4.00)")
    st.sidebar.subheader("1. Data Infrastructure")
    d1_q1 = st.sidebar.slider(INDICATOR_MAP["D1_Data_Quality"], 1.0, 4.0, 2.5, 0.01)
    d1_q2 = st.sidebar.slider(INDICATOR_MAP["D1_Data_Governance"], 1.0, 4.0, 2.5, 0.01)
    d1_q3 = st.sidebar.slider(INDICATOR_MAP["D1_Data_Integration"], 1.0, 4.0, 2.5, 0.01)
    
    st.sidebar.subheader("2. Technological Maturity")
    d2_q1 = st.sidebar.slider(INDICATOR_MAP["D2_System_Capability"], 1.0, 4.0, 2.5, 0.01)
    d2_q2 = st.sidebar.slider(INDICATOR_MAP["D2_AI_Tooling"], 1.0, 4.0, 2.5, 0.01)
    d2_q3 = st.sidebar.slider(INDICATOR_MAP["D2_Infrastructure_Resilience"], 1.0, 4.0, 2.5, 0.01)
    
    st.sidebar.subheader("3. Regulatory Compliance")
    d3_q1 = st.sidebar.slider(INDICATOR_MAP["D3_FCA_Alignment"], 1.0, 4.0, 2.5, 0.01)
    d3_q2 = st.sidebar.slider(INDICATOR_MAP["D3_Consumer_Duty"], 1.0, 4.0, 2.5, 0.01)
    d3_q3 = st.sidebar.slider(INDICATOR_MAP["D3_Audit_Trail"], 1.0, 4.0, 2.5, 0.01)
    
    st.sidebar.subheader("4. Organisational Capability")
    d4_q1 = st.sidebar.slider(INDICATOR_MAP["D4_Talent_Readiness"], 1.0, 4.0, 2.5, 0.01)
    d4_q2 = st.sidebar.slider(INDICATOR_MAP["D4_Change_Management"], 1.0, 4.0, 2.5, 0.01)
    d4_q3 = st.sidebar.slider(INDICATOR_MAP["D4_Leadership_Commitment"], 1.0, 4.0, 2.5, 0.01)
    
    st.sidebar.subheader("5. Ethical Governance")
    d5_q1 = st.sidebar.slider(INDICATOR_MAP["D5_Bias_Mitigation"], 1.0, 4.0, 2.5, 0.01)
    d5_q2 = st.sidebar.slider(INDICATOR_MAP["D5_Explainability"], 1.0, 4.0, 2.5, 0.01)
    d5_q3 = st.sidebar.slider(INDICATOR_MAP["D5_Accountability"], 1.0, 4.0, 2.5, 0.01)
    
    d1_avg = np.mean([d1_q1, d1_q2, d1_q3])
    d2_avg = np.mean([d2_q1, d2_q2, d2_q3])
    d3_avg = np.mean([d3_q1, d3_q2, d3_q3])
    d4_avg = np.mean([d4_q1, d4_q2, d4_q3])
    d5_avg = np.mean([d5_q1, d5_q2, d5_q3])
    
    st.markdown("### AIRI Dynamic Sensitivity Weighting Structure")
    st.write("Adjust the raw slider allocations below to automatically normalize and evaluate alternative risk structures.")
    
    col_w1, col_w2, col_w3, col_w4, col_w5 = st.columns(5)
    with col_w1:
        w1 = st.slider("Data Weight", 0.0, 1.0, 0.20, 0.01)
    with col_w2:
        w2 = st.slider("Technology Weight", 0.0, 1.0, 0.20, 0.01)
    with col_w3:
        w3 = st.slider("Governance Weight", 0.0, 1.0, 0.20, 0.01)
    with col_w4:
        w4 = st.slider("Organisation Weight", 0.0, 1.0, 0.20, 0.01)
    with col_w5:
        w5 = st.slider("Ethics Weight", 0.0, 1.0, 0.20, 0.01)
        
    total_w = w1 + w2 + w3 + w4 + w5
    if total_w == 0:
        total_w = 1.0
        
    weights = {
        "D1": w1 / total_w,
        "D2": w2 / total_w,
        "D3": w3 / total_w,
        "D4": w4 / total_w,
        "D5": w5 / total_w
    }
        
    weight_df = pd.DataFrame({
        "Dimension": [
            "Data Infrastructure",
            "Technological Maturity",
            "Regulatory Compliance",
            "Organisational Capability",
            "Ethical Governance"
        ],
        "Normalized Active Weight": [
            weights["D1"], 
            weights["D2"], 
            weights["D3"], 
            weights["D4"], 
            weights["D5"]
        ]
    })
    
    st.dataframe(
        weight_df, 
        use_container_width=True,
        column_config={
            "Normalized Active Weight": st.column_config.NumberColumn(format="%.3f")
        }
    )
    
    raw_composite = (
        d1_avg * weights["D1"] +
        d2_avg * weights["D2"] +
        d3_avg * weights["D3"] +
        d4_avg * weights["D4"] +
        d5_avg * weights["D5"]
    )
    composite_score_100 = ((raw_composite - 1.0) / 3.0) * 100
    
    if not all(np.isclose(v, 0.20, atol=1e-2) for v in [weights["D1"], weights["D2"], weights["D3"], weights["D4"], weights["D5"]]):
        st.warning("""
        **Custom Sensitivity Scenario Activated:**
        The metric card and graph below reflect your custom weight allocation.
        """)
        
    col_metrics, col_chart = st.columns([1, 2])
    with col_metrics:
        st.metric(
            label="Calculated Composite AIRI Score",
            value=f"{composite_score_100:.2f}%"
        )
        
        if composite_score_100 >= 76:
            band = "Advanced 🚀"
        elif composite_score_100 >= 51:
            band = "Established ✅"
        elif composite_score_100 >= 26:
            band = "Developing ⚠️"
        else:
            band = "Nascent 🚨"
            
        st.success(f"Deterministic Category: {band}")
        
        st.write("---")
        st.markdown("##### ⚙️ Exploratory Predictive Classification Output")
        
        if model_loaded:
            input_dict = {
                "D1_Data_Quality": [d1_q1], "D1_Data_Governance": [d1_q2], "D1_Data_Integration": [d1_q3],
                "D2_System_Capability": [d2_q1], "D2_AI_Tooling": [d2_q2], "D2_Infrastructure_Resilience": [d2_q3],
                "D3_FCA_Alignment": [d3_q1], "D3_Consumer_Duty": [d3_q2], "D3_Audit_Trail": [d3_q3],
                "D4_Talent_Readiness": [d4_q1], "D4_Change_Management": [d4_q2], "D4_Leadership_Commitment": [d4_q3],
                "D5_Bias_Mitigation": [d5_q1], "D5_Explainability": [d5_q2], "D5_Accountability": [d5_q3]
            }
            try:
                missing_cols = set(feature_cols) - set(input_dict.keys())
                if missing_cols:
                    st.error(f"Missing required model inputs: {missing_cols}")
                else:
                    input_df = pd.DataFrame(input_dict)[feature_cols]
                    ml_pred = rf_model.predict(input_df)[0]
                    ml_prob = rf_model.predict_proba(input_df)[0]
                    classes = rf_model.classes_
                    
                    readiness_mapping = {0: "Nascent 🚨", 1: "Developing ⚠️", 2: "Established ✅", 3: "Advanced 🚀"}
                    ml_label = readiness_mapping.get(ml_pred, str(ml_pred))
                    
                    st.info(f"Classifier Inference: **{ml_label}**")
                    st.info("""
                    The sensitivity weighting structure affects only the deterministic
                    AIRI composite calculation. The machine learning classifier operates
                    independent using the raw institutional indicator inputs.
                    """)
                    
                    st.write("**Prediction Probability Distribution:**")
                    prob_df = pd.DataFrame({"Class": classes, "Probability": ml_prob}).sort_values(by="Probability", ascending=False)
                    
                    for _, row in prob_df.iterrows():
                        cls_labeled = readiness_mapping.get(row["Class"], str(row["Class"]))
                        st.caption(f"{cls_labeled}: {row['Probability']*100:.1f}%")
            except Exception as e:
                st.error(f"Model inference error: {e}")
        else:
            st.warning("ML artifacts not found. Running in deterministic scoring mode only.")
            
    with col_chart:
        performance_df = pd.DataFrame({
            "Dimension": [
                "Data Infrastructure",
                "Technological Maturity",
                "Regulatory Compliance",
                "Organisational Capability",
                "Ethical Governance"
            ],
            "Performance (%)": [
                ((d1_avg - 1) / 3) * 100,
                ((d2_avg - 1) / 3) * 100,
                ((d3_avg - 1) / 3) * 100,
                ((d4_avg - 1) / 3) * 100,
                ((d5_avg - 1) / 3) * 100
            ]
        })

        weighted_df = pd.DataFrame({
            "Dimension": performance_df["Dimension"],
            "Weighted Contribution (%)": [
                performance_df["Performance (%)"][0] * weights["D1"],
                performance_df["Performance (%)"][1] * weights["D2"],
                performance_df["Performance (%)"][2] * weights["D3"],
                performance_df["Performance (%)"][3] * weights["D4"],
                performance_df["Performance (%)"][4] * weights["D5"]
            ]
        })

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.barplot(
            x="Performance (%)",
            y="Dimension",
            data=performance_df,
            ax=axes[0],
            palette="Blues_r"
        )
        axes[0].set_title("AIRI Dimension Performance (Unweighted)")
        axes[0].set_xlim(0, 100)
        axes[0].axvline(50, color="grey", linestyle="--", alpha=0.4)
        axes[0].set_xlabel("Performance Score (%)")
        axes[0].set_ylabel("")

        sns.barplot(
            x="Weighted Contribution (%)",
            y="Dimension",
            data=weighted_df,
            ax=axes[1],
            palette="viridis"
        )
        axes[1].set_title("AIRI Weighted Contribution (Policy Adjusted)")
        axes[1].set_xlim(0, max(weighted_df["Weighted Contribution (%)"]) + 5)
        axes[1].axvline(100 / 5, color="red", linestyle="--", alpha=0.6)
        axes[1].set_xlabel("Weighted Impact (%)")
        axes[1].set_ylabel("")

        plt.suptitle(
            "AIRI Dual-Mode Governance View: Performance vs Weighted Impact",
            fontsize=14,
            fontweight="bold"
        )

        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.write("---")

    st.markdown("### Real-Time Local Explainability (current institution)")
    st.write("""
    This section provides an institution-specific prediction explanation utilizing SHAP metrics 
    derived from the active input configurations simulated above.
    """)

    if model_loaded and explainer is not None:
        try:
            active_input_dict = {
                "D1_Data_Quality": [d1_q1],
                "D1_Data_Governance": [d1_q2],
                "D1_Data_Integration": [d1_q3],
                "D2_System_Capability": [d2_q1],
                "D2_AI_Tooling": [d2_q2],
                "D2_Infrastructure_Resilience": [d2_q3],
                "D3_FCA_Alignment": [d3_q1],
                "D3_Consumer_Duty": [d3_q2],
                "D3_Audit_Trail": [d3_q3],
                "D4_Talent_Readiness": [d4_q1],
                "D4_Change_Management": [d4_q2],
                "D4_Leadership_Commitment": [d4_q3],
                "D5_Bias_Mitigation": [d5_q1],
                "D5_Explainability": [d5_q2],
                "D5_Accountability": [d5_q3]
            }

            missing_shap_cols = set(feature_cols) - set(active_input_dict.keys())

            if missing_shap_cols:
                st.error(
                    f"Missing required features for SHAP generation: {missing_shap_cols}"
                )
            else:
                runtime_df = pd.DataFrame(active_input_dict)[feature_cols]
                shap_values = explainer(runtime_df)

                if hasattr(shap_values, "values") and isinstance(shap_values.values, np.ndarray):
                    predicted_class_index = int(
                        rf_model.predict(runtime_df)[0]
                    )

                    shap_array = shap_values.values

                    if len(shap_array.shape) == 3:
                        if shap_array.shape[1] == len(runtime_df.columns):
                            class_shap_values = shap_array[0, :, predicted_class_index]
                        else:
                            class_shap_values = shap_array[0, predicted_class_index, :]
                    else:
                        class_shap_values = shap_array[0]

                    feature_names = runtime_df.columns.tolist()
                    shap_vals = class_shap_values

                    shap_df = pd.DataFrame({
                        "Feature": feature_names,
                        "SHAP Value": shap_vals
                    })

                    shap_df = shap_df.sort_values(by="SHAP Value", key=abs)

                    fig, ax = plt.subplots(figsize=(9, 5))

                    bars = ax.barh(
                        shap_df["Feature"].str.replace("_", " "),
                        shap_df["SHAP Value"]
                    )

                    ax.axvline(0, color="black", linewidth=1, alpha=0.6)
                    ax.set_title(
                        "Live Institutional SHAP Feature Contribution",
                        fontsize=13,
                        pad=12
                    )

                    ax.set_xlabel("SHAP Impact on Prediction")
                    ax.set_ylabel("Governance Indicators")

                    for bar in bars:
                        width = bar.get_width()
                        ax.text(
                            width,
                            bar.get_y() + bar.get_height() / 2,
                            f"{width:.2f}",
                            va="center",
                            fontsize=8
                        )

                    plt.tight_layout()
                    st.pyplot(fig, clear_figure=True)
                else:
                    st.warning(
                        "Unsupported SHAP explanation format detected."
                    )
        except Exception as shap_err:
            st.error(
                f"Live SHAP visualization compilation exception encountered: "
                f"{shap_err}"
            )
    else:
        st.info(
            "Model artifacts must be fully running to initiate "
            "live SHAP interpretation charts."
        )
        
    st.write("---")
    st.markdown("### 📋 Prescriptive Remediation & Strategic Guidance")
    col_rem1, col_rem2 = st.columns(2)
    with col_rem1:
        if d3_q2 < 3.0:
            st.error(f"⚠️ Critical Consumer Duty Deficit ({INDICATOR_MAP['D3_Consumer_Duty']}): The institution demonstrates insufficient auditing for consumer outcomes. Prioritise stronger monitoring and governance controls.")
        if d5_q1 < 3.0:
            st.error(f"⚠️ Ethical Risk Alert ({INDICATOR_MAP['D5_Bias_Mitigation']}): Bias mitigation mechanisms appear underdeveloped relative to expected governance standards.")
    with col_rem2:
        if d1_q2 < 3.0:
            st.warning(f"💡 Data Infrastructure Advisory ({INDICATOR_MAP['D1_Data_Governance']}): Data lineage and provenance mapping structures appear weak. Consider implementing clearer governance tracing procedures.")
        if d4_q1 >= 3.0 and d2_q2 < 2.5:
            st.info("💡 Operational Balance Note: Organisational readiness appears stronger than automated tooling maturity. Additional workflow automation may improve alignment.")

with tab_performance:
    st.warning("""
    These performance metrics are exploratory and derived from synthetic,
    expert-guided simulation data rather than operational institutional datasets.
    The outputs should therefore be interpreted as proof-of-concept validation
    rather than production-grade predictive performance.
    """)

    st.markdown("### 🔬 Exploratory Readiness Classification Model Metrics")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("#### 🌲 Random Forest Classifier (Selected Model)")
        rf_metrics_df = pd.DataFrame({
            "Evaluation Parameter": ["Global Accuracy", "Weighted F1-Score", "Macro F1-Score", "Cohen's Kappa", "Root Mean Squared Error (RMSE)"],
            "Value": ["0.940", "0.939", "0.860", "0.892", "0.245"]
        })
        st.dataframe(rf_metrics_df, use_container_width=True)
        st.markdown("##### Classification Performance Breakdown")
        class_df = pd.DataFrame([
            {"Maturity Tier": "Nascent", "Precision": 1.00, "Recall": 0.50, "F1-Score": 0.67, "Support": 2},
            {"Maturity Tier": "Developing", "Precision": 0.89, "Recall": 0.89, "F1-Score": 0.89, "Support": 9},
            {"Maturity Tier": "Established", "Precision": 0.91, "Recall": 0.94, "F1-Score": 0.92, "Support": 31},
            {"Maturity Tier": "Advanced", "Precision": 0.97, "Recall": 0.97, "F1-Score": 0.97, "Support": 58}
        ])
        st.dataframe(class_df, use_container_width=True)
    with col_m2:
        st.markdown("#### ⚡ XGBoost Classifier (Baseline Model)")
        xgb_metrics_df = pd.DataFrame({
            "Evaluation Parameter": ["Global Accuracy", "Weighted F1-Score", "Macro F1-Score", "Cohen's Kappa", "Root Mean Squared Error (RMSE)"],
            "Value": ["0.900", "0.896", "0.779", "0.817", "0.316"]
        })
        st.dataframe(xgb_metrics_df, use_container_width=True)
        st.markdown("##### Illustrative Feature Importance Distribution")
        feat_imp_df = pd.DataFrame([
            {"Feature Component": "D1_Data_Quality", "Random Forest Contribution": 0.1078, "XGBoost Contribution": 0.0517},
            {"Feature Component": "D5_Bias_Mitigation", "Random Forest Contribution": 0.0923, "XGBoost Contribution": 0.2155},
            {"Feature Component": "D3_Consumer_Duty", "Random Forest Contribution": 0.0867, "XGBoost Contribution": 0.0631},
            {"Feature Component": "D4_Talent_Readiness", "Random Forest Contribution": 0.0861, "XGBoost Contribution": 0.0654},
            {"Feature Component": "D4_Change_Management", "Random Forest Contribution": 0.0759, "XGBoost Contribution": 0.0574}
        ])
        fig_imp, ax_imp = plt.subplots(figsize=(6, 3))
        sns.barplot(x="Random Forest Contribution", y="Feature Component", data=feat_imp_df, palette="magma", ax=ax_imp)
        plt.title("Illustrative Feature Importance Ranking")
        plt.tight_layout()
        st.pyplot(fig_imp, clear_figure=True)
        
    st.write("---")
    st.markdown("### Model-Wide Feature Attribution (trained behaviour overview)")
    try:
        st.image("shap_summary_plot.png", caption="SHAP Model-Wide Feature Importance Summary Matrix (Static Baseline Framework)")
    except Exception:
        st.info("Static global SHAP summary graphic asset could not be accessed at this time.")

with tab_empirical:
    st.markdown("### 📈 Pre-Deployment Verification Analysis of Expert Panels")
    col_e1, col_e2, col_e3 = st.columns(3)
    col_e1.metric("Fleiss' Kappa Reliability", "0.97")
    col_e2.metric("S-CVI / Average Score", "0.934")
    col_e3.metric("Indicators Cronbach Alpha", "0.91")
    st.markdown("#### Content Validity Index (I-CVI) Matrix")
    cvi_df = pd.DataFrame([
        {"Indicator ID": "IND-D1-01", "Description": "Data Quality Systematic Monitoring", "I-CVI": 0.833, "Mean Relevance": 3.317},
        {"Indicator ID": "IND-D1-02", "Description": "Data Stewardship & Lineage Tracking", "I-CVI": 0.914, "Mean Relevance": 3.362},
        {"Indicator ID": "IND-D1-03", "Description": "Integrated Systems Architecture", "I-CVI": 0.879, "Mean Relevance": 3.353},
        {"Indicator ID": "IND-D2-01", "Description": "Machine Learning Deployment Capability", "I-CVI": 0.930, "Mean Relevance": 3.548},
        {"Indicator ID": "IND-D2-02", "Description": "MLOps Lifecycle & Drift Governance", "I-CVI": 0.905, "Mean Relevance": 3.500},
        {"Indicator ID": "IND-D2-03", "Description": "Infrastructure Resilience & Recovery", "I-CVI": 0.930, "Mean Relevance": 3.623},
        {"Indicator ID": "IND-D3-01", "Description": "Documented FCA Alignment Standards", "I-CVI": 0.923, "Mean Relevance": 3.632},
        {"Indicator ID": "IND-D3-02", "Description": "FCA Consumer Duty Outcome Auditing", "I-CVI": 0.950, "Mean Relevance": 3.650},
        {"Indicator ID": "IND-D3-03", "Description": "Immutable Outcome Auditing Logs", "I-CVI": 0.940, "Mean Relevance": 3.650},
        {"Indicator ID": "IND-D4-01", "Description": "Talent Readiness & System Literacy", "I-CVI": 0.974, "Mean Relevance": 3.687},
        {"Indicator ID": "IND-D4-02", "Description": "Structured Change Control Controls", "I-CVI": 0.966, "Mean Relevance": 3.701},
        {"Indicator ID": "IND-D4-03", "Description": "Executive Budget Sponsorship Ownership", "I-CVI": 0.974, "Mean Relevance": 3.741},
        {"Indicator ID": "IND-D5-01", "Description": "Fairness Assessment & Bias Mitigation", "I-CVI": 0.983, "Mean Relevance": 3.741},
        {"Indicator ID": "IND-D5-02", "Description": "Customer-Facing Explainability Protocols", "I-CVI": 0.966, "Mean Relevance": 3.735},
        {"Indicator ID": "IND-D5-03", "Description": "Clear Accountability Oversight Structures", "I-CVI": 0.949, "Mean Relevance": 3.632}
    ])
    st.dataframe(cvi_df, use_container_width=True)

with tab_feedback:
    st.markdown("### 🧪 Stage 5 Expert Interaction & Evaluation")
    st.text_input("Expert ID (optional)", key="expert_id")
    st.markdown("## System Usability Scale (SUS) – AIRI Adapted")
    
    sus_q = {}
    sus_questions = [
        "I think I would like to use the AIRI dashboard frequently in my organisation.",
        "I found the AIRI dashboard unnecessarily complex.",
        "I thought the dashboard was easy to use.",
        "I think I would need technical support to use this system.",
        "I found the various functions in the dashboard well integrated.",
        "I thought there was too much inconsistency in the dashboard.",
        "I imagine most professionals would learn to use the AIRI dashboard quickly.",
        "I found the dashboard very cumbersome to use.",
        "I felt confident using the AIRI governance dashboard.",
        "I needed to learn a lot before I could use the system."
    ]
    for i, q in enumerate(sus_questions, start=1):
        sus_q[i] = st.radio(f"SUS {i}. {q}", [1, 2, 3, 4, 5], horizontal=True, index=2, key=f"sus_radio_{i}")
        
    sus_score = 0
    for i in range(1, 11):
        if i % 2 == 1:
            sus_score += sus_q[i] - 1
        else:
            sus_score += 5 - sus_q[i]
    sus_total = sus_score * 2.5
    st.metric("System Usability Scale (SUS) Score", f"{sus_total:.2f} / 100")
    
    if sus_total >= 80:
        st.success("Excellent perceived usability")
    elif sus_total >= 68:
        st.info("Above average usability")
    else:
        st.warning("Usability may require refinement")
        
    st.markdown("---")
    st.markdown("## Thematic Evaluation (Expert Reflection)")
    t1 = st.text_area("1. How easy was it to understand the AIRI dashboard and its outputs?")
    t2 = st.text_area("2. Which component was most useful (sliders, ML prediction, score, charts) and why?")
    t3 = st.text_area("3. Did the AIRI score and ML prediction align with your expectations? Explain.")
    t4 = st.text_area("4. How useful is the dashboard for governance or compliance decisions (e.g. FCA alignment, Consumer Duty)?")
    t5 = st.text_area("5. What limitations did you observe for real-world institutional use?")
    t6 = st.text_area("6. How confident are you in the ML-generated predictions?")
    t7 = st.text_area("7. What would improve your trust or understanding of the system?")
    
    if st.button("Save Expert Feedback"):
        feedback_df = pd.DataFrame([{
            "Expert_ID": st.session_state.get("expert_id", "anonymous"), "SUS_Score": sus_total,
            "Q1": sus_q[1], "Q2": sus_q[2], "Q3": sus_q[3], "Q4": sus_q[4], "Q5": sus_q[5],
            "Q6": sus_q[6], "Q7": sus_q[7], "Q8": sus_q[8], "Q9": sus_q[9], "Q10": sus_q[10],
            "T1": t1, "T2": t2, "T3": t3, "T4": t4, "T5": t5, "T6": t6, "T7": t7
        }])

        file_path = "airi_expert_feedback_master.csv"
        with lock:
            if os.path.exists(file_path):
                feedback_df.to_csv(file_path, mode="a", header=False, index=False)
            else:
                feedback_df.to_csv(file_path, index=False)
        st.success("Feedback saved successfully!")

    st.write("---")
    with st.expander("🔐 Admin View: Review & Manage Master Feedback"):
        admin_password = st.text_input(
            "Enter Admin Password to access data management tools",
            type="password"
        )

        ADMIN_PASSWORD = os.getenv("AIRI_ADMIN_PASSWORD")

        # -------------------------------
        # ACCESS CONTROL LOGIC (FIXED)
        # -------------------------------
        if ADMIN_PASSWORD is None:
            st.error("Admin password is not configured in environment variables.")

        elif admin_password and admin_password == ADMIN_PASSWORD:
            st.success("Access granted.")

            file_path = "airi_expert_feedback_master.csv"

            if os.path.exists(file_path):
                master_df = pd.read_csv(file_path)

                st.markdown(f"**Total Expert Responses Collected:** `{len(master_df)}`")

                if len(master_df) > 0:
                    st.metric(
                        "Average Evaluated SUS Score",
                        f"{master_df['SUS_Score'].mean():.2f} / 100"
                    )

                st.dataframe(master_df, use_container_width=True)

                st.write("---")
                st.markdown("#### 🗑️ Delete Feedback Entries")

                if len(master_df) > 0:
                    row_to_delete = st.number_input(
                        "Enter the Row Index number you want to remove:",
                        min_value=0,
                        max_value=len(master_df)-1,
                        step=1
                    )

                    st.warning(
                        f"Target row preview: Index `{row_to_delete}` | "
                        f"Expert ID: `{master_df.iloc[row_to_delete].get('Expert_ID', 'anonymous')}`"
                    )

                    if st.checkbox("I confirm that I want to permanently delete this row."):
                        if st.button("🔴 Permanently Delete Selected Row", type="primary"):
                            with lock:
                                master_df = master_df.drop(master_df.index[row_to_delete])
                                master_df.to_csv(file_path, index=False)
                            st.success("Row deleted successfully. Refresh app.")

        else:
            # Only show error if user actually tried
            if admin_password:
                st.error("Incorrect password.")
