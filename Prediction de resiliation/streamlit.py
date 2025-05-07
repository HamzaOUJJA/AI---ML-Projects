# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib

# Charger le modèle
model = joblib.load('rf.pkl')  # modèle RandomForest entraîné

st.title("Prédiction de résiliation d'abonnement")

# Entrées utilisateur
age = st.slider("Âge", 18, 100, 30)
revenu = st.number_input("Revenu mensuel", value=3000)
sexe = st.selectbox("Sexe", ["Homme", "Femme"])
anciennete = st.slider("Ancienneté (en années)", 0, 10, 2)
frequence = st.slider("Fréquence d'utilisation (mois)", 0, 30, 10)
support = st.selectbox("A-t-il contacté le support ?", ["Oui", "Non"])
satisfaction = st.slider("Score de satisfaction (1-10)", 1, 10, 5)

# ----------- Prétraitements manuels pour correspondre aux features -------------

# Sexe encodé : 1 = Homme, 0 = Femme → colonne 'Sexe_1'
sexe_1 = 1 if sexe == "Homme" else 0

# Support_contacte_1 : 1 si Oui, 0 sinon
support_1 = 1 if support == "Oui" else 0

# Satisfaction en catégories one-hot
satisfaction_bon = 1 if satisfaction >= 8 else 0
satisfaction_moyen = 1 if 5 <= satisfaction < 8 else 0
# (on déduit le niveau "Faible" par absence des deux autres)

# Age groupé
if age >= 70:
    age_tres_senior = 1
    age_senior = 0
    age_adulte = 0
elif age >= 50:
    age_tres_senior = 0
    age_senior = 1
    age_adulte = 0
elif age >= 30:
    age_tres_senior = 0
    age_senior = 0
    age_adulte = 1
else:
    age_tres_senior = 0
    age_senior = 0
    age_adulte = 0  # en dessous de 30 : catégorie non modélisée ?

# High_value_customer (exemple : seuil revenu > 5000)
high_value = 1 if revenu >= 5000 else 0

# Fréquence par an (si frequence = tous les X mois, donc 12 / X)
freq_per_year = round(12 / frequence, 2) if frequence > 0 else 0

# ----------- Créer le DataFrame avec les bonnes colonnes -----------------------

input_data = pd.DataFrame([{
    'Satisfaction_catégorie_Bon': satisfaction_bon,
    'Satisfaction_catégorie_Moyen': satisfaction_moyen,
    'Age_groupe_Très Senior': age_tres_senior,
    'Age_groupe_Senior': age_senior,
    'Age_groupe_Adulte': age_adulte,
    'High_value_customer': high_value,
    'Support_contacte_1': support_1,
    'Age': age,
    'Sexe_1': sexe_1,
    'Frequence_utilisation': frequence,
    'Anciennete': anciennete,
    'Freq_per_year': freq_per_year
}])

# ---------------- Prédiction ----------------
if st.button("Prédire"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Risque élevé de résiliation ({proba:.2%})")
    else:
        st.success(f"✅ Client fidèle (Risque faible : {proba:.2%})")
