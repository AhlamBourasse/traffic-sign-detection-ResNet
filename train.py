import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

# Charger les étiquettes à partir du fichier labels.csv
labels_df = pd.read_csv('labels.csv')
class_names = labels_df.set_index('ClassId')['Name'].to_dict()

# Définir les chemins vers les données
data_dir = 'myData'
train_dir = 'myData/train'
test_dir = 'myData/test'

# Créer les répertoires train et test pour chaque classe
for class_id in range(1, 8):
    class_name = class_names[class_id]
    os.makedirs(os.path.join(train_dir, str(class_id)), exist_ok=True)
    os.makedirs(os.path.join(test_dir, str(class_id)), exist_ok=True)

# Diviser les données en ensembles d'entraînement et de test pour chaque classe
for class_id in range(1, 8):
    class_dir = os.path.join(data_dir, str(class_id))
    images = os.listdir(class_dir)
    train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)
    
    for img in train_images:
        src = os.path.join(class_dir, img)
        new_img_name = img.replace(':', '-')  # Remplacer les caractères spéciaux dans le nom de fichier
        dst = os.path.join(train_dir, str(class_id), new_img_name)
        shutil.copy(src, dst)
    
    for img in test_images:
        src = os.path.join(class_dir, img)
        new_img_name = img.replace(':', '-')  # Remplacer les caractères spéciaux dans le nom de fichier
        dst = os.path.join(test_dir, str(class_id), new_img_name)
        shutil.copy(src, dst)

# Transformation des données
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Charger les données
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

# Vérifier si l'accélération GPU est disponible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Utiliser l'architecture ResNet pré-entraînée
model = models.resnet18(pretrained=True)
# Remplacer la couche de classification finale pour correspondre au nombre de classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 7)

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Scheduler pour ajuster le taux d'apprentissage
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Entraînement du modèle
model.to(device)
for epoch in range(10):
    print('Epoch {}/{}'.format(epoch, 9))
    print('-' * 10)

    for phase in ['train', 'test']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

# Sauvegarder le modèle
torch.save(model.state_dict(), 'model_resnet.pth')

# Afficher l'accuracy après l'entraînement
print('Train Accuracy: {:.4f}'.format(epoch_acc))
