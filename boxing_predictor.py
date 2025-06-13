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

# 🧠 Vorhersagefunktion
def predict_theoretical_match(boxer1, boxer2, df, model, feature_names):
    boxer_a, boxer_b = boxer1, boxer2

    # Feature-Mittelwerte berechnen
    b1_data = df[(df["opponent_1_name_clean"] == boxer_a) | (df["opponent_2_name_clean"] == boxer_a)].mean(numeric_only=True)
    b2_data = df[(df["opponent_1_name_clean"] == boxer_b) | (df["opponent_2_name_clean"] == boxer_b)].mean(numeric_only=True)

    if b1_data.empty or b2_data.empty:
        st.error("❌ Für einen oder beide Boxer konnten keine ausreichenden Daten gefunden werden.")
        return

    # Eingabe-Datenframe aufbauen
    input_data = {}
    for feat in feature_names:
        if "opponent_1" in feat:
            base_feat = feat.replace("opponent_1_", "")
            input_data[feat] = b1_data.get(f"opponent_1_{base_feat}", b1_data.get(base_feat, 0))
        elif "opponent_2" in feat:
            base_feat = feat.replace("opponent_2_", "")
            input_data[feat] = b2_data.get(f"opponent_2_{base_feat}", b2_data.get(base_feat, 0))

    input_df = pd.DataFrame([input_data])

    # Sicherstellen, dass alle Features vorhanden sind
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_names]

    # Vorhersage
    probs = model.predict_proba(input_df)[0]
    pred = model.predict(input_df)[0]

    # Ausgabe
    st.markdown("### 📈 Gewinnwahrscheinlichkeiten:")
    st.write(f"**{boxer_a} gewinnt:** {probs[1]*100:.2f}%")
    st.write(f"**{boxer_b} gewinnt:** {probs[0]*100:.2f}%")

    sieger = boxer_a if pred == 1 else boxer_b
    st.markdown(f"### 🏆 Erwarteter Sieger: **{sieger}**")

# 🧪 Button
if st.button("🔮 Vorhersage starten"):
    predict_theoretical_match(boxer1, boxer2, df, model, feature_names)

