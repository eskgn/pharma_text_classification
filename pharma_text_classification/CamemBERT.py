from data import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from transformers import CamembertTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from multiprocessing import freeze_support
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

class PharmaDataset(Dataset): # Pour PyTorch DataLoader
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ------------------ 3. Tout le code principal ------------------
if __name__ == "__main__":

    # Tokenization
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    train_encodings = tokenizer(
        train_df['text'].tolist(),
        truncation=True,
        padding=True,
        max_length=32
    )
    val_encodings = tokenizer(
        val_df['text'].tolist(),
        truncation=True,
        padding=True,
        max_length=32
    )

    train_labels = train_df['label_encoded'].values
    val_labels = val_df['label_encoded'].values

    # Création des DataLoaders
    train_dataset = PharmaDataset(train_encodings, train_labels)
    val_dataset = PharmaDataset(val_encodings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)  # Windows-friendly
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=0)

    # ------------------ 4. Modèle ------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        "camembert-base",
        num_labels=num_classes
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max', 
    factor=0.7,
    patience=2,
    )

    # ------------------ 5. Entraînement ------------------
    epochs = 3
    # Stocker les métriques par epoch
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []
    train_f1_scores = []  # Pour stocker le F1-score d'entraînement

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        all_train_preds, all_train_labels = [], []  # Pour calculer le F1-score d'entraînement

        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            # Calcul des prédictions pour le F1-score d'entraînement
            with torch.no_grad():
                preds = torch.argmax(outputs.logits, dim=1)
                all_train_preds.extend(preds.cpu().numpy())
                all_train_labels.extend(labels.cpu().numpy())

        # Calcul des métriques d'entraînement
        epoch_train_loss = total_loss/len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Calcul du F1-score pour l'entraînement
        train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')
        train_f1_scores.append(train_f1)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        all_val_preds, all_val_labels = [], []
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Calcul de la perte pendant la validation
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss = outputs.loss
                total_val_loss += val_loss.item()
                
                # Prédictions
                preds = torch.argmax(outputs.logits, dim=1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        
        # Calcul des métriques de validation
        epoch_val_loss = total_val_loss/len(val_loader)  # Perte moyenne de validation
        val_losses.append(epoch_val_loss)
        acc = accuracy_score(all_val_labels, all_val_preds)
        f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
        val_accuracies.append(acc)
        val_f1_scores.append(f1)

        print(f"Validation Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
        scheduler.step(f1)

# ------------------ 6. Prédictions finales ------------------
model.eval()
preds = []
true_labels = []  # Stocker les vrais labels

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        batch_preds = torch.argmax(outputs.logits, dim=1)
        preds.extend(batch_preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Conversion numérique -> texte
preds_classes = le.inverse_transform(preds)
true_classes = le.inverse_transform(true_labels)

# ------------------ 7. Évaluation détaillée ------------------
# 1. Rapport de classification
print("\n" + "="*50)
print("Rapport de classification détaillé:")
print("="*50)

# Utiliser les noms de classes texte
report = classification_report(
    true_classes, 
    preds_classes,
    digits=4,
    zero_division=0
)
print(report)

# 2. Matrice de confusion
conf_matrix = confusion_matrix(true_classes, preds_classes)

# Obtenir les noms de classes texte uniques
unique_classes = sorted(set(true_classes) | set(preds_classes))

plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=unique_classes,
    yticklabels=unique_classes
)
plt.title('Matrice de Confusion')
plt.xlabel('Prédictions')
plt.ylabel('Vraies étiquettes')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# 3. Analyse par classe
print("\n" + "="*50)
print("Récapitulatif par classe:")
print("="*50)

# Utiliser les classes texte pour l'analyse
class_counts = pd.Series(true_classes).value_counts()
class_metrics = []

for class_name in unique_classes:
    # Vérifier si la classe existe dans les prédictions/vérités
    if class_name in true_classes or class_name in preds_classes:
        # Calculer TP, FP, FN
        tp = conf_matrix[unique_classes.index(class_name), unique_classes.index(class_name)]
        fn = sum(conf_matrix[unique_classes.index(class_name), :]) - tp
        fp = sum(conf_matrix[:, unique_classes.index(class_name)]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics.append({
            'Classe': class_name,
            'Exemples': class_counts.get(class_name, 0),
            'Précision': f"{precision:.4f}",
            'Rappel': f"{recall:.4f}",
            'F1': f"{f1:.4f}"
        })

# ------------------ 8. Graphique des métrics par epoch ------------------
plt.figure(figsize=(12, 5))

# Graphique 1: Évolution des pertes
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_losses, 'o-', color='blue', label='Train')
plt.plot(range(1, epochs+1), val_losses, 'o-', color='red', label='Validation')
plt.title('Évolution des pertes')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(1, epochs+1))
plt.grid(True, alpha=0.3)
plt.legend()

# Graphique 2: Évolution des F1-scores
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), train_f1_scores, 'o-', color='orange', label='Train')
plt.plot(range(1, epochs+1), val_f1_scores, 'o-', color='green', label='Validation')
plt.title('Évolution des F1-scores')
plt.xlabel('Epoch')
plt.ylabel('F1-score')
plt.xticks(range(1, epochs+1))
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics.png', dpi=300)
plt.show()

metrics_df = pd.DataFrame(class_metrics)
print(metrics_df.to_string(index=False))

