import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torchvision import models
import torch.nn as nn
import pandas as pd

# Charger les étiquettes à partir du fichier labels.csv
labels_df = pd.read_csv('labels.csv')
class_names = labels_df.set_index('ClassId')['Name'].to_dict()

# Charger le modèle pré-entraîné
model = models.resnet18(weights=None)  # Utilisez 'weights=None' au lieu de 'pretrained=False'
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)  # Utilisez 'nn.Linear' au lieu de 'np.Linear'

# Charger les poids du modèle entraîné
model.load_state_dict(torch.load('model_resnet.pth', map_location=torch.device('cpu')))  # Charger le modèle sur le CPU
model.eval()  # Mettre le modèle en mode évaluation

# Transformer l'image capturée pour correspondre aux transformations utilisées lors de l'entraînement
image_transforms = transforms.Compose([
    transforms.ToPILImage(),  # Ajout de cette transformation pour travailler avec des images PIL
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Fonction pour prédire la classe de l'image
def predict_image(image):
    image = image_transforms(image)  # Appliquer les transformations à l'image
    image = image.unsqueeze(0)  # Ajouter une dimension de batch
    with torch.no_grad():  # Pas besoin de calculer les gradients lors de la prédiction
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()  # Retourne la prédiction sous forme de nombre entier

# Capturer les images de la webcam en temps réel
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Redimensionner la frame capturée à 150x170 pixels
    frame = cv2.resize(frame, (150, 170))

    # Afficher la frame capturée
    cv2.imshow('Webcam', frame)

    # Détecter les touches 'q' ou 'Q' pour quitter
    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        break

    # Traiter l'image et faire une prédiction
    prediction = predict_image(frame)
    class_name = class_names[prediction + 1]  # Ajouter 1 car les classes sont indexées à partir de 1
    print("Predicted class:", class_name)

# Libérer la webcam et fermer les fenêtres OpenCV
cap.release()
cv2.destroyAllWindows()
