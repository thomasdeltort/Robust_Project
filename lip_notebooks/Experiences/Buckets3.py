import torch
import boto3
from botocore.exceptions import ClientError
import io

# --- Configuration ---
# Remplacez par le nom de votre bucket et le chemin exact de votre modèle
BUCKET_NAME = 'tdrobustbucket'
MODEL_KEY = 'lip_models/FC_2MOONS_Lip.pt.pt' # ⚠️ Mettez le nom correct de votre fichier .pt

# --- Chargement du modèle ---
print(f"Tentative de chargement du modèle '{MODEL_KEY}' depuis le bucket '{BUCKET_NAME}'...")

# Initialiser le client S3
s3 = boto3.client('s3')

try:
    # Obtenir l'objet depuis S3
    response = s3.get_object(Bucket=BUCKET_NAME, Key=MODEL_KEY)
    
    # Lire le contenu du fichier (le corps de la réponse) dans un buffer en mémoire
    model_data = io.BytesIO(response['Body'].read())
    
    # Charger le modèle directement depuis le buffer
    # Utiliser map_location pour s'assurer que le modèle se charge sur le CPU si aucun GPU n'est dispo
    model = torch.load(model_data, map_location=torch.device('cpu'))
    
    # Mettre le modèle en mode évaluation (important pour l'inférence)
    model.eval()
    
    print("✅ Modèle chargé avec succès !")
    # Afficher l'architecture du modèle pour confirmer
    print(model)

except ClientError as e:
    if e.response['Error']['Code'] == 'NoSuchKey':
        print(f"❌ ERREUR : Le fichier '{MODEL_KEY}' n'a pas été trouvé dans le bucket '{BUCKET_NAME}'.")
    else:
        print(f"❌ ERREUR S3 inattendue : {e}")
except Exception as e:
    print(f"❌ ERREUR lors du chargement du modèle avec PyTorch : {e}")