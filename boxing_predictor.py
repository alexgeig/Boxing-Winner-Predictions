import streamlit as st
import pandas as pd
import joblib
import numpy as np
from xgboost import XGBClassifier

# 🔃 Modell und Feature-Namen laden
model = joblib.load("xgb_model.pkl")
feature_names = joblib.load("feature_names.pkl")
df = pd.read_csv("merged_df_encoded.csv")

# 🎯 GUI-Titel
st.title("🥊 Boxing Match Predictor")
st.markdown("Wähle zwei Boxer aus und erhalte eine **theoretische Vorhersage** über den Kampf-Ausgang.")

# 📋 Boxerliste
boxer_names = sorted(set(df["opponent_1_name_clean"].dropna()) | set(df["opponent_2_name_clean"].dropna()))

# 📥 Eingabe durch den Nutzer
boxer1 = st.selectbox("👤 Boxer 1", boxer_names)
boxer2 = st.selectbox("👤 Boxer 2", [n for n in boxer_names if n != boxer1])

# 🧠 Vorhersagefunktion mit symmetrischer Vorhersage
def predict_proba_for(b1, b2):
    input_data = {}
    for feat in feature_names:
        val = np.nan
        if "opponent_1" in feat:
            base_feat = feat.replace("opponent_1_", "")
            col1 = f"opponent_1_{base_feat}"
            col2 = f"opponent_2_{base_feat}"
            if col1 in df.columns:
                val = df[df["opponent_1_name_clean"] == b1][col1].mean()
            if np.isnan(val) and col2 in df.columns:
                val = df[df["opponent_2_name_clean"] == b1][col2].mean()
            input_data[feat] = val if not np.isnan(val) else 0
        else:
            base_feat = feat.replace("opponent_2_", "")
            col1 = f"opponent_1_{base_feat}"
            col2 = f"opponent_2_{base_feat}"
            if col1 in df.columns:
                val = df[df["opponent_1_name_clean"] == b2][col1].mean()
            if np.isnan(val) and col2 in df.columns:
                val = df[df["opponent_2_name_clean"] == b2][col2].mean()
            input_data[feat] = val if not np.isnan(val) else 0

    input_df = pd.DataFrame([input_data])[feature_names].fillna(0)
    return model.predict_proba(input_df)[0]

def predict_theoretical_match(boxer1, boxer2):
    boxer_a, boxer_b = boxer1, boxer2

    probs_a_first = predict_proba_for(boxer_a, boxer_b)
    probs_b_first = predict_proba_for(boxer_b, boxer_a)

    prob_boxer_a = (probs_a_first[1] + probs_b_first[0]) / 2
    prob_boxer_b = (probs_a_first[0] + probs_b_first[1]) / 2

    st.markdown("### 📈 Gewinnwahrscheinlichkeiten:")
    st.write(f"**{boxer_a} gewinnt:** {prob_boxer_a * 100:.2f}%")
    st.write(f"**{boxer_b} gewinnt:** {prob_boxer_b * 100:.2f}%")

    sieger = boxer_a if prob_boxer_a > prob_boxer_b else boxer_b
    st.markdown(f"### 🏆 Erwarteter Sieger: **{sieger}**")

# 🧪 Button
if st.button("🔮 Vorhersage starten"):
    predict_theoretical_match(boxer1, boxer2)
