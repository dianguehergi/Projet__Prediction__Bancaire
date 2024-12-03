import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataPreparation:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        if os.path.exists(self.file_path):
            self.data = pd.read_csv(self.file_path, sep=';')
        else:
            raise FileNotFoundError(f"Le fichier {self.file_path} est introuvable. Veuillez vérifier le chemin.")

    def exploratory_analysis(self):
        print("Aperçu des 5 premières lignes des données :")
        print(self.data.head())

        print("\nInformations sur les données :")
        print(self.data.info())

        print("\nStatistiques descriptives :")
        print(self.data.describe())

    def prepare_data(self):
        # Encodage de la variable cible
        self.y = self.data['y'].apply(lambda x: 1 if x == 'yes' else 0)

        # Séparation des caractéristiques et de la variable cible
        self.X = self.data.drop('y', axis=1)

        # Séparation des variables numériques et catégorielles
        numeric_features = self.X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.X.select_dtypes(include=['object']).columns

        # Pipeline de transformation des données
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first')

        # Application des transformations aux colonnes appropriées
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        # Transformation des données
        self.X = preprocessor.fit_transform(self.X)

        # Division des données en ensembles d'entraînement et de test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

    def show_data_summary(self):
        summary = {
            'X_train shape': self.X_train.shape,
            'X_test shape': self.X_test.shape,
            'y_train distribution': pd.Series(self.y_train).value_counts().to_dict(),
            'y_test distribution': pd.Series(self.y_test).value_counts().to_dict()
        }
        summary_df = pd.DataFrame.from_dict(summary, orient='index', columns=['Value'])
        print("\nRésumé des données préparées :")
        print(summary_df)

    def visualize_data(self):
        # Distribution de la variable cible
        plt.figure(figsize=(6, 4))
        sns.countplot(x='y', data=self.data)
        plt.title("Distribution de la variable cible (souscription)")
        plt.xlabel("Souscription (y)")
        plt.ylabel("Nombre de clients")
        plt.show()

        # Analyse des variables numériques
        plt.figure(figsize=(10, 8))
        self.data.select_dtypes(include=['int64', 'float64']).hist(bins=15, figsize=(15, 10), layout=(3, 3))
        plt.suptitle("Distribution des variables numériques")
        plt.show()

        # Analyse des variables catégorielles
        categorical_features = self.data.select_dtypes(include=['object']).columns
        for feature in categorical_features:
            plt.figure(figsize=(10, 4))
            sns.countplot(y=feature, data=self.data, order=self.data[feature].value_counts().index)
            plt.title(f"Distribution de la variable catégorielle : {feature}")
            plt.xlabel("Nombre de clients")
            plt.ylabel(feature)
            plt.show()

        # Analyse des corrélations
        plt.figure(figsize=(12, 10))
        corr_matrix = self.data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title("Matrice de corrélation des variables numériques")
        plt.show()

        # Analyse des relations entre certaines variables et la variable cible
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='y', y='age', data=self.data)
        plt.title("Relation entre l'âge et la souscription")
        plt.xlabel("Souscription (y)")
        plt.ylabel("Age")
        plt.show()

# Utilisation de la classe
file_path = r"C:\Users\hergi\Documents\hergiDiangue\PGE3Fr\Coding for AI & Data Science\projetS1\Data\bank-additional\bank-additional\bank-additional-full.csv"
data_prep = DataPreparation(file_path)
data_prep.load_data()
data_prep.exploratory_analysis()
data_prep.prepare_data()
data_prep.show_data_summary()
data_prep.visualize_data()
