import streamlit as st
import base64

# Charger l'image et la convertir en base64
def charger_image_en_base64(chemin_image):
    with open(chemin_image, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

def appliquer_css():
    """
    Fonction pour appliquer du CSS personnalisé à l'application Streamlit.
    """
    # Chemin de l'image
    chemin_image = r"C:\Users\hergi\Documents\hergiDiangue\PGE3Fr\Coding for AI & Data Science\projetS1\image\Bank1.jpg"
    image_base64 = charger_image_en_base64(chemin_image)

    st.markdown(
        f"""
        <style>
            /* Personnalisation générale de l'application */
            .stApp {{
                background: url('data:image/jpeg;base64,{image_base64}') no-repeat center center fixed;
                background-size: cover;
            }}

            .main {{
                background-color: rgba(255, 255, 255, 0.8);
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
            }}

            /* En-tête principal stylisé */
            h1 {{
                color: #003366;
                text-align: center;
                font-family: 'Verdana', sans-serif;
                font-size: 2.5em;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}

            /* Barre latérale (Entrées Utilisateur) */
            .sidebar .sidebar-content {{
                background: #f0f5f9;
                border-radius: 15px;
                padding: 15px;
                border: 2px solid #004c99;
            }}

            .stSidebar .st-ae .st-ax {{
                background: #004c99;
                border-radius: 8px;
                border: 1px solid #004c99;
                padding: 10px;
                margin-bottom: 15px;
            }}

            .st-ae .st-ax select {{
                background-color: #f7fafc;
                border: 1px solid #004c99;
                border-radius: 4px;
                padding: 8px;
            }}

            .stSidebar h3 {{
                color: #004080;
                font-weight: bold;
            }}

            /* Boutons Streamlit */
            .stButton>button {{
                background-color: #007acc;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                text-align: center;
                text-decoration: none;
                font-size: 18px;
                font-weight: bold;
                margin: 10px 5px;
                cursor: pointer;
                transition: background-color 0.3s, transform 0.3s;
            }}

            .stButton>button:hover {{
                background-color: #005f99;
                transform: scale(1.05);
            }}

            /* Widgets de la barre latérale */
            .css-1lcbmhc {{
                background-color: #004c99;
                padding: 10px;
                border-radius: 12px;
                border: 2px solid #003366;
                box-shadow: 0px 0px 10px rgba(0, 0, 128, 0.2);
            }}

            .css-1v0mbdj {{
                color: #003366;
                font-weight: bold;
            }}

            /* Section Tableau de Bord */
            .dashboard-section {{
                background-color: #f4f9ff;
                border-radius: 10px;
                padding: 20px;
                margin-top: 20px;
                border: 1px solid #004080;
                box-shadow: 0px 0px 10px rgba(0, 64, 128, 0.1);
            }}

            /* Graphiques */
            .stPlotlyChart {{
                background-color: rgba(255, 255, 255, 0.9);
                border-radius: 15px;
                box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.15);
                padding: 15px;
            }}

            /* Personnalisation des éléments de texte */
            p, label, h2, h3 {{
                color: #004080;
                font-family: 'Arial', sans-serif;
            }}

            /* Inputs personnalisés */
            input[type="number"], input[type="text"] {{
                background-color: #eef7ff;
                border: 2px solid #007acc;
                border-radius: 5px;
                padding: 10px;
                font-size: 16px;
                margin-bottom: 10px;
                width: 100%;
            }}

            input[type="number"]:focus, input[type="text"]:focus {{
                border-color: #005f99;
                outline: none;
                box-shadow: 0 0 5px #007acc;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

def appliquer_header():
    """
    Fonction pour appliquer un en-tête stylisé en HTML.
    """
    st.markdown(
        """
        <div style="background-color: #0066cc; padding: 15px; border-radius: 10px; box-shadow: 0px 0px 15px rgba(0, 0, 128, 0.3);">
            <h1 style="color: white; text-align: center; font-family: 'Verdana', sans-serif;">Tableau de Bord de Prédiction de Souscription Bancaire</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# Appeler ces fonctions dans le script principal pour appliquer le CSS personnalisé et l'en-tête
appliquer_css()
appliquer_header()
