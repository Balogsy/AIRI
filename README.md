# AIRI Dashboard Prototype

## Artificial Intelligence Readiness Index (AIRI) for Financial Institutions

The AIRI Dashboard Prototype is an interactive machine learning application designed to assess the AI readiness of financial institutions. The system combines synthetic data generation, machine learning, explainable AI, and visual analytics to support AI readiness evaluation and decision-making.

The dashboard measures readiness across key areas including data, technology, governance, organisation, and ethical AI practices. Using validated AIRI indicators, the platform calculates institutional readiness scores and predicts readiness categories such as Nascent, Developing, Established, and Advanced.

The prototype was developed using Python, Streamlit, Random Forest, XGBoost, SHAP, and SDV. A synthetic dataset containing 500 financial institutions was generated to simulate realistic AI readiness patterns for analysis and model training.

The dashboard includes interactive assessment sliders, AIRI score calculation, machine learning prediction, feature importance analysis, SHAP explainability, radar chart visualisation, sensitivity analysis, and downloadable assessment outputs.

This project demonstrates how machine learning and explainable analytics can support AI governance and readiness assessment within the financial sector.

# Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- SDV

# Project Structure
AIRI_Dashboard/
│
├── app.py
├── rf_airi_model.pkl
├── feature_columns.pkl
├── requirements.txt
└── README.md
