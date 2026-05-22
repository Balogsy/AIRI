

# ============================================================
# AIRI STREAMLIT DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AIRI Dashboard",
    layout="wide"
)

# ============================================================
# LOAD MODEL
# ============================================================

with open("rf_airi_model.pkl", "rb") as file:
    rf_model = pickle.load(file)

with open("feature_columns.pkl", "rb") as file:
    feature_columns = pickle.load(file)

# ============================================================
# TITLE
# ============================================================

st.title("AIRI Readiness Assessment Dashboard")

st.markdown("""
Artificial Intelligence Readiness Index (AIRI)
for Financial Institutions
""")

# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.header("Institution Assessment Inputs")

input_values = {}

for feature in feature_columns:

    input_values[feature] = st.sidebar.slider(

        feature,

        min_value=1.0,
        max_value=4.0,
        value=3.0,
        step=0.01
    )

# ============================================================
# SENSITIVITY ANALYSIS WEIGHTS
# ============================================================

st.sidebar.header("Sensitivity Analysis: Adjust Dimension Weights")

w1 = st.sidebar.slider(
    "Weight - Data Dimension",
    min_value=0.0,
    max_value=1.0,
    value=0.20,
    step=0.01
)

w2 = st.sidebar.slider(
    "Weight - Technology Dimension",
    min_value=0.0,
    max_value=1.0,
    value=0.20,
    step=0.01
)

w3 = st.sidebar.slider(
    "Weight - Governance Dimension",
    min_value=0.0,
    max_value=1.0,
    value=0.20,
    step=0.01
)

w4 = st.sidebar.slider(
    "Weight - Organisation Dimension",
    min_value=0.0,
    max_value=1.0,
    value=0.20,
    step=0.01
)

w5 = st.sidebar.slider(
    "Weight - Ethics Dimension",
    min_value=0.0,
    max_value=1.0,
    value=0.20,
    step=0.01
)

# ============================================================
# NORMALISE WEIGHTS
# ============================================================

total_weight = w1 + w2 + w3 + w4 + w5

w1 = w1 / total_weight
w2 = w2 / total_weight
w3 = w3 / total_weight
w4 = w4 / total_weight
w5 = w5 / total_weight

# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_df = pd.DataFrame(
    [input_values]
)

# ============================================================
# CALCULATE DIMENSION SCORES
# ============================================================

D1 = np.mean([
    input_values["D1_Data_Quality"],
    input_values["D1_Data_Governance"],
    input_values["D1_Data_Integration"]
])

D2 = np.mean([
    input_values["D2_System_Capability"],
    input_values["D2_AI_Tooling"],
    input_values["D2_Infrastructure_Resilience"]
])

D3 = np.mean([
    input_values["D3_FCA_Alignment"],
    input_values["D3_Consumer_Duty"],
    input_values["D3_Audit_Trail"]
])

D4 = np.mean([
    input_values["D4_Talent_Readiness"],
    input_values["D4_Change_Management"],
    input_values["D4_Leadership_Commitment"]
])

D5 = np.mean([
    input_values["D5_Bias_Mitigation"],
    input_values["D5_Explainability"],
    input_values["D5_Accountability"]
])

# ============================================================
# AIRI SCORE
# ============================================================

airi_score = (

    D1 * w1 +
    D2 * w2 +
    D3 * w3 +
    D4 * w4 +
    D5 * w5
)

airi_score_100 = round(
    ((airi_score - 1) / 3) * 100,
    2
)

# ============================================================
# DETERMINISTIC READINESS TIER
# ============================================================

if airi_score_100 <= 25:

    deterministic_tier = "Nascent"

elif airi_score_100 <= 50:

    deterministic_tier = "Developing"

elif airi_score_100 <= 75:

    deterministic_tier = "Established"

else:

    deterministic_tier = "Advanced"

# ============================================================
# MACHINE LEARNING PREDICTION
# ============================================================

prediction = rf_model.predict(input_df)[0]

readiness_mapping = {

    0: "Nascent",
    1: "Developing",
    2: "Established",
    3: "Advanced"
}

predicted_readiness = readiness_mapping[prediction]

# ============================================================
# DISPLAY RESULTS
# ============================================================

st.header("Assessment Results")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "AIRI Score",
        f"{airi_score_100}"
    )

with col2:

    st.metric(
        "Rule-Based Tier",
        deterministic_tier
    )

with col3:

    st.metric(
        "ML Predicted Tier",
        predicted_readiness
    )

# ============================================================
# DISPLAY WEIGHTS
# ============================================================

st.subheader("Current Dimension Weights")

weights_df = pd.DataFrame({

    "Dimension": [
        "Data",
        "Technology",
        "Governance",
        "Organisation",
        "Ethics"
    ],

    "Weight": [
        round(w1, 3),
        round(w2, 3),
        round(w3, 3),
        round(w4, 3),
        round(w5, 3)
    ]
})

st.dataframe(weights_df)

# ============================================================
# RADAR CHART
# ============================================================

st.subheader("Dimension Radar Chart")

categories = [
    "Data",
    "Technology",
    "Governance",
    "Organisation",
    "Ethics"
]

values = [D1, D2, D3, D4, D5]

values += values[:1]

angles = np.linspace(
    0,
    2 * np.pi,
    len(categories),
    endpoint=False
).tolist()

angles += angles[:1]

fig, ax = plt.subplots(
    figsize=(6,6),
    subplot_kw=dict(polar=True)
)

ax.plot(
    angles,
    values,
    linewidth=2
)

ax.fill(
    angles,
    values,
    alpha=0.25
)

ax.set_xticks(angles[:-1])

ax.set_xticklabels(categories)

ax.set_title("AIRI Dimension Performance")

st.pyplot(fig)

# ============================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

st.subheader("Random Forest Feature Importance")

importance_df = pd.DataFrame({

    "Feature": feature_columns,
    "Importance": rf_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

st.dataframe(importance_df)

# ============================================================
# FEATURE IMPORTANCE BAR CHART
# ============================================================

fig2, ax2 = plt.subplots(figsize=(10,6))

ax2.barh(

    importance_df["Feature"],
    importance_df["Importance"]
)

ax2.invert_yaxis()

ax2.set_xlabel("Importance")

ax2.set_title(
    "Random Forest Feature Importance"
)

st.pyplot(fig2)

# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

st.subheader("SHAP Explainability Analysis")

explainer = shap.TreeExplainer(rf_model)

shap_values = explainer.shap_values(input_df)

fig3, ax3 = plt.subplots(figsize=(8,4))

if isinstance(shap_values, list):

    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value[0],
        shap_values[0][0],
        feature_names=feature_columns,
        show=False
    )

else:

    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value,
        shap_values[0],
        feature_names=feature_columns,
        show=False
    )

st.pyplot(fig3)

# ============================================================
# DIMENSION SCORES TABLE
# ============================================================

st.subheader("Dimension Scores")

results_df = pd.DataFrame({

    "Dimension": categories,

    "Score": [
        round(D1, 2),
        round(D2, 2),
        round(D3, 2),
        round(D4, 2),
        round(D5, 2)
    ]
})

st.dataframe(results_df)

# ============================================================
# DOWNLOAD RESULTS
# ============================================================

st.subheader("Download Results")

csv = results_df.to_csv(index=False)

st.download_button(

    label="Download Assessment CSV",

    data=csv,

    file_name="AIRI_Assessment.csv",

    mime="text/csv"
)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
AIRI Dashboard Prototype

Developed using Streamlit + Machine Learning + SHAP Explainability
""")

