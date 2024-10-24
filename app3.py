import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.cluster import KMeans

# Fonction principale pour exécuter l'application
def main():
    st.title("Application de Prédiction de Segmentation et Churn des Clients")

    # Formulaire pour entrer les données du client
    st.subheader("Entrez les données du client")
    
    # Saisie des données
    order_amount_hike = st.number_input("Augmentation du montant de la commande par rapport à l'année dernière", min_value=0.0)
    order_count = st.number_input("Nombre de commandes", min_value=0)
    hour_spend_on_app = st.number_input("Heures passées sur l'application", min_value=0.0)
    coupon_used = st.number_input("Coupon utilisé (0 pour Non, 1 pour Oui)", min_value=0, max_value=1)
    cashback_amount = st.number_input("Montant du cashback", min_value=0.0)
    satisfaction_score = st.number_input("Score de satisfaction (0-10)", min_value=0, max_value=10)
    gender = st.selectbox("Genre", ["Female", "Male"])
    tenure = st.number_input("Durée en mois comme client", min_value=0)

    if st.button("Prédire la Segmentation et le Churn"):
        # Conversion du genre
        gender_numeric = 0 if gender == "Female" else 1

        # Créer un DataFrame à partir des données saisies
        client_data = pd.DataFrame({
            'OrderAmountHikeFromlastYear': [order_amount_hike],
            'OrderCount': [order_count],
            'HourSpendOnApp': [hour_spend_on_app],
            'CouponUsed': [coupon_used],
            'CashbackAmount': [cashback_amount],
            'SatisfactionScore': [satisfaction_score],
            'Gender': [gender_numeric],
            'Tenure': [tenure]
        })

        # Prédiction de la segmentation
        segment = predict_segmentation(client_data)
        st.write(f"Segment du client: {segment}")

        # Prédiction du churn
        churn_prediction = predict_churn(client_data)
        st.write(f"Le client est susceptible de se désabonner: {'Oui' if churn_prediction else 'Non'}")

def predict_segmentation(client_data):
    # Charger le dataset pour l'entraînement
    df = pd.read_csv('e_commerce_dataset.csv')
    df = preprocess_data(df)  # Prétraitement des données pour s'assurer qu'il n'y a pas de NaN
    
    # Variables pour la segmentation
    variables_segmentation = ['OrderAmountHikeFromlastYear', 'OrderCount', 'HourSpendOnApp', 'CouponUsed', 'CashbackAmount']
    
    # Normalisation sur l'ensemble du dataset
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[variables_segmentation])
    
    # Application de K-means avec 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(df_scaled)  # Entraîner le modèle sur l'ensemble des données
    
    # Normaliser les données du client
    client_data_scaled = scaler.transform(client_data[variables_segmentation])
    
    # Prédire le cluster du client
    cluster = kmeans.predict(client_data_scaled)

    # Définir les segments
    if cluster[0] == 0:
        return 'Clients VIP'
    elif cluster[0] == 1:
        return 'Clients fidèles mais inactifs'
    elif cluster[0] == 2:
        return 'Clients à faible engagement'
    else:
        return 'Clients occasionnels'

def predict_churn(client_data):
    # Charger un modèle déjà entraîné
    model_choice = 'Régression Logistique'  # Choisissez le modèle ici pour la prédiction de churn
    features = ['OrderAmountHikeFromlastYear', 'OrderCount', 'HourSpendOnApp', 'CouponUsed', 'CashbackAmount', 'SatisfactionScore', 'Gender', 'Tenure']

    # Chargement du dataset pour l'entraînement
    df = pd.read_csv('e_commerce_dataset.csv')
    df = preprocess_data(df)

    # Définir X et y
    X = df[features]
    y = df['Churn']

    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Entraînement du modèle
    if model_choice == 'Régression Logistique':
        model = LogisticRegression(random_state=42)
    elif model_choice == 'Forêt Aléatoire':
        model = RandomForestClassifier(random_state=42)
    elif model_choice == 'XGBoost':
        model = xgb.XGBClassifier(random_state=42)
    
    model.fit(X_train_scaled, y_train)

    # Préparation des données du client pour la prédiction
    client_data_scaled = scaler.transform(client_data[features])
    
    # Prédire le churn
    churn_prediction = model.predict(client_data_scaled)
    
    return churn_prediction[0]  # Retourner True (1) ou False (0)

def preprocess_data(df):
    # Séparation des colonnes numériques et catégorielles
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Remplissage des colonnes numériques avec la médiane
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

    # Remplissage des colonnes catégorielles avec le mode
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Conversion des variables catégorielles en numériques
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

    return df

if __name__ == "__main__":
    main()
