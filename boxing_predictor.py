import streamlit as st
import pandas as pd
import joblib
import numpy as np
from xgboost import XGBClassifier

# 🔃 Modell, LabelEncoder und Feature-Namen laden
model = XGBClassifier()
model.load_model("xgb_model.json")
le = joblib.load("label_encoder.pkl")
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
def predict_theoretical_match(boxer1, boxer2, df, model, le, feature_names):
    boxer_a, boxer_b = sorted([boxer1, boxer2])

    # Feature-Mittelwerte berechnen
    b1_data = df[(df["opponent_1_name_clean"] == boxer_a) | (df["opponent_2_name_clean"] == boxer_a)].mean(numeric_only=True)
    b2_data = df[(df["opponent_1_name_clean"] == boxer_b) | (df["opponent_2_name_clean"] == boxer_b)].mean(numeric_only=True)

    if b1_data.empty or b2_data.empty:
        st.error("❌ Für einen oder beide Boxer konnten keine ausreichenden Daten gefunden werden.")
        return

    # Eingabe-Datenframe aufbauen
    input_data = {}
    for feat in feature_names:
        if "_1" in feat:
            base_feat = feat.replace("opponent_1", "opponent")
            input_data[feat] = b1_data.get(base_feat, 0)
        elif "_2" in feat:
            base_feat = feat.replace("opponent_2", "opponent")
            input_data[feat] = b2_data.get(base_feat, 0)

    input_df = pd.DataFrame([input_data])[feature_names].fillna(0)

    # Vorhersage
    probs = model.predict_proba(input_df)[0]
    pred = model.predict(input_df)[0]
    pred_label = le.inverse_transform([pred])[0]

    # Siegername ermitteln
    if pred_label == "WON":
        sieger = boxer_a
    elif pred_label == "LOSS":
        sieger = boxer_b
    else:
        sieger = "Unentschieden"

    # Ausgabe
    st.markdown("### 📈 Gewinnwahrscheinlichkeiten:")
    for i, label in enumerate(le.classes_):
        st.write(f"**{label}**: {probs[i]*100:.2f}%")

    st.markdown(f"### 🏆 Erwartetes Ergebnis: **{sieger}** ")

# 🧪 Button
if st.button("🔮 Vorhersage starten"):
    predict_theoretical_match(boxer1, boxer2, df, model, le, feature_names)
