import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split



from streamlit_design import appliquer_css, appliquer_header

appliquer_css()
appliquer_header()



# Paramètres de style pour les graphiques
sns.set(style="whitegrid")

# Charger le pipeline du meilleur modèle
model_path = "best_model_pipeline.pkl"

if not os.path.exists(model_path):
    st.error("Le fichier nécessaire pour le modèle n'est pas disponible. Assurez-vous que 'best_model_pipeline.pkl' est présent dans le répertoire.")
else:
    # Charger le pipeline enregistré
    best_pipeline = joblib.load(model_path)

    # Charger les données pour générer des métriques réalistes
    data_path = r"C:\Users\hergi\Documents\hergiDiangue\PGE3Fr\Coding for AI & Data Science\projetS1\Data\bank\bank.csv"
    if not os.path.exists(data_path):
        st.error("Le fichier de données 'bank.csv' n'est pas disponible pour générer des métriques réalistes.")
    else:
        data = pd.read_csv(data_path, sep=';')
        X = data.drop(columns=['y'])
        y = data['y'].apply(lambda x: 1 if x == 'yes' else 0)
        
        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Titre de l'application
        #st.title("Tableau de Bord de Prédiction de Souscription Bancaire")

        # Saisie des caractéristiques par l'utilisateur
        st.sidebar.header("Entrées Utilisateur")
        age = st.sidebar.number_input("Âge", min_value=18, max_value=100, value=30)
        job = st.sidebar.selectbox("Type d'emploi", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
        marital = st.sidebar.selectbox("État civil", ['married', 'single', 'divorced'])
        education = st.sidebar.selectbox("Niveau d'éducation", ['primary', 'secondary', 'tertiary', 'unknown'])
        default = st.sidebar.selectbox("Crédit en défaut ?", ['no', 'yes'])
        balance = st.sidebar.number_input("Solde du compte bancaire", value=1000)
        housing = st.sidebar.selectbox("Prêt immobilier ?", ['no', 'yes'])
        loan = st.sidebar.selectbox("Prêt personnel ?", ['no', 'yes'])
        contact = st.sidebar.selectbox("Type de contact", ['cellular', 'telephone', 'unknown'])
        day = st.sidebar.number_input("Jour de contact (1-31)", min_value=1, max_value=31, value=15)
        month = st.sidebar.selectbox("Mois de contact", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
        duration = st.sidebar.number_input("Durée du contact (secondes)", value=100)
        campaign = st.sidebar.number_input("Nombre de contacts durant cette campagne", value=1)
        pdays = st.sidebar.number_input("Jours depuis le dernier contact (999 si aucun contact)", value=999)
        previous = st.sidebar.number_input("Nombre de contacts avant cette campagne", value=0)
        poutcome = st.sidebar.selectbox("Résultat de la campagne précédente", ['unknown', 'failure', 'success', 'other'])

        # Possibilité de choisir un seuil de décision
        threshold = st.sidebar.slider("Seuil de décision pour la prédiction", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

        # Création du DataFrame pour les données entrées
        input_data = pd.DataFrame([{
            'age': age,
            'job': job,
            'marital': marital,
            'education': education,
            'default': default,
            'balance': balance,
            'housing': housing,
            'loan': loan,
            'contact': contact,
            'day': day,
            'month': month,
            'duration': duration,
            'campaign': campaign,
            'pdays': pdays,
            'previous': previous,
            'poutcome': poutcome
        }])

        # Prédiction
        if st.button("Prédire"):
            # Utiliser le pipeline pour prétraiter et prédire
            probability = best_pipeline.predict_proba(input_data)[0][1]  # Probabilité de souscription
            result = 'Souscription' if probability >= threshold else 'Pas de souscription'
            st.subheader("Résultat de la Prédiction")
            st.write(f"Prédiction : {result}")
            st.write(f"Probabilité de souscription : {np.round(probability * 100, 2)}%")

            # Générer des prédictions réalistes pour l'ensemble de test
            y_pred_proba_test = best_pipeline.predict_proba(X_test)[:, 1]
            y_pred_test = (y_pred_proba_test >= threshold).astype(int)

            # Tracer la courbe ROC
            st.subheader("Courbe ROC")
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
            roc_auc = auc(fpr, tpr)

            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, color='blue', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.2f})')
            ax_roc.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
            ax_roc.set_xlabel('Taux de Faux Positifs')
            ax_roc.set_ylabel('Taux de Vrais Positifs')
            ax_roc.set_title('Courbe ROC')
            ax_roc.legend(loc='lower right')
            st.pyplot(fig_roc)

            # Afficher la matrice de confusion
            st.subheader("Matrice de Confusion")
            cm = confusion_matrix(y_test, y_pred_test)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_xlabel('Prédictions')
            ax_cm.set_ylabel('Vérités')
            ax_cm.set_title('Matrice de Confusion')
            st.pyplot(fig_cm)

            # Scores de performance
            st.subheader("Scores de Performance")
            accuracy = accuracy_score(y_test, y_pred_test)
            precision = precision_score(y_test, y_pred_test, zero_division=1)
            recall = recall_score(y_test, y_pred_test, zero_division=1)
            f1 = f1_score(y_test, y_pred_test, zero_division=1)
            st.write(f"Exactitude (Accuracy) : {accuracy:.2f}")
            st.write(f"Précision (Precision) : {precision:.2f}")
            st.write(f"Rappel (Recall) : {recall:.2f}")
            st.write(f"Score F1 : {f1:.2f}")

            # Calcul du Bénéfice Net Business
            st.subheader("Bénéfice Net Business")
            Gain_VP = st.sidebar.number_input("Gain par client ayant souscrit (VP)", value=1000)
            Gain_VN = st.sidebar.number_input("Gain pour éviter une tentative infructueuse (VN)", value=100)
            Coût_FP = st.sidebar.number_input("Coût d'une tentative infructueuse (FP)", value=50)
            Coût_FN = st.sidebar.number_input("Coût d'une opportunité manquée (FN)", value=500)

            VP = cm[1, 1]
            VN = cm[0, 0]
            FP = cm[0, 1]
            FN = cm[1, 0]

            Bénéfice_Net = (VP * Gain_VP) + (VN * Gain_VN) - (FP * Coût_FP) - (FN * Coût_FN)
            st.write(f"Bénéfice Net : {Bénéfice_Net} €")

            # Comparaison des Modèles
            st.subheader("Comparaison des Modèles")
            model_names = ["Logistic Regression", "Random Forest", "Gradient Boosting", "Support Vector Machine"]
            model_accuracies = [0.84, 0.87, 0.85, 0.83]  # Exemple de valeurs de précision pour chaque modèle
            
            fig_compare, ax_compare = plt.subplots()
            sns.barplot(x=model_names, y=model_accuracies, ax=ax_compare, palette='viridis')
            ax_compare.set_title("Comparaison des Modèles - Précision")
            ax_compare.set_ylabel("Précision")
            ax_compare.set_xlabel("Modèles")
            ax_compare.set_ylim(0.0, 1.0)
            st.pyplot(fig_compare)

            # Analyse des Groupes de Clients
            st.subheader("Analyse des Groupes de Clients")
            age_group = st.sidebar.slider("Sélectionnez la tranche d'âge", min_value=18, max_value=100, value=(25, 50))
            selected_group = data[(data['age'] >= age_group[0]) & (data['age'] <= age_group[1])]
            group_mean_prob = best_pipeline.predict_proba(selected_group.drop(columns=['y']))[:, 1].mean()
            st.write(f"Probabilité moyenne de souscription pour la tranche d'âge {age_group[0]}-{age_group[1]} : {group_mean_prob:.2f}")

            # Simulation de Scénarios
            st.subheader("Simulation de Scénarios")
            simulation_balance = st.sidebar.slider("Modifier le solde du compte bancaire pour la simulation", min_value=-5000, max_value=50000, value=1000, step=500)
            input_data_simulation = input_data.copy()
            input_data_simulation['balance'] = simulation_balance
            simulation_probability = best_pipeline.predict_proba(input_data_simulation)[0][1]
            st.write(f"Probabilité de souscription après modification du solde à {simulation_balance}€ : {simulation_probability:.2f}")

        # Informations supplémentaires
        st.write("\n**Note :** Les résultats de la prédiction sont basés sur les caractéristiques entrées et sur le modèle de machine learning entraîné. Veuillez utiliser ces informations avec discernement.")
