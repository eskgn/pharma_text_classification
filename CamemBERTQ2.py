from dataQ2 import *
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from multiprocessing import freeze_support
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.manifold import TSNE
import torch.nn.functional as F
import umap.umap_ as umap
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
import os
from pathlib import Path

# ==================== CONFIGURATION ====================
MODEL_DIR = Path("./saved_model")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "camembert_finetuned"
TRAINING_COMPLETE_FLAG = MODEL_DIR / "training_complete.txt"

# Palette de 11 couleurs distinctes (GLOBALE pour cohérence)
COLORS_11 = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
    '#008080'
]

def get_color_mapping(labels, le):
    """Créer un mapping cohérent classe -> couleur"""
    unique_labels_sorted = sorted(np.unique(labels))
    label_to_color = {}
    for i, lab in enumerate(unique_labels_sorted):
        lab_name = str(le.inverse_transform([lab])[0])
        label_to_color[lab_name] = COLORS_11[i % 11]
    return label_to_color

class PharmaDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def save_model(model, tokenizer, optimizer, epoch, metrics):
    """Sauvegarde complète du modèle et des métadonnées"""
    print(f"\n💾 Sauvegarde du modèle dans {MODEL_PATH}...")
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, MODEL_PATH / "training_state.pt")
    
    with open(TRAINING_COMPLETE_FLAG, 'w') as f:
        f.write(f"Training completed at epoch {epoch}")
    print("✅ Modèle sauvegardé avec succès!")

def load_model(num_classes, device):
    """Charge le modèle sauvegardé s'il existe"""
    if MODEL_PATH.exists() and TRAINING_COMPLETE_FLAG.exists():
        print(f"\n📂 Chargement du modèle existant depuis {MODEL_PATH}...")
        tokenizer = CamembertTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        
        training_state = torch.load(MODEL_PATH / "training_state.pt")
        print(f"✅ Modèle chargé (entraîné jusqu'à l'epoch {training_state['epoch']})")
        return model, tokenizer, training_state
    return None, None, None

def create_enhanced_umap_2d(embeddings, labels, le, filename="umap_2d_enhanced.png"):
    """Visualisation UMAP 2D propre sans erreurs"""
    X = embeddings.astype(np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    
    N = X.shape[0]
    n_neighbors = max(15, min(100, int(np.sqrt(N))))
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1,
                       metric='cosine', random_state=42, n_jobs=1)
    z2 = reducer.fit_transform(X)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 14), facecolor='white')
    
    # Utiliser le mapping de couleurs cohérent
    label_to_color = get_color_mapping(labels, le)
    unique_labels = sorted(np.unique(labels))
    
    for lab in unique_labels:
        mask = labels == lab
        if mask.sum() == 0:
            continue
        
        xi, yi = z2[mask, 0], z2[mask, 1]
        lab_name = str(le.inverse_transform([lab])[0])
        color = label_to_color[lab_name]
        
        ax.scatter(xi, yi, s=80, alpha=0.8, marker='o', 
                  edgecolor='black', linewidth=0.5, color=color,
                  label=lab_name, zorder=3)
    
    ax.set_title("UMAP 2D - Espace latent des 11 classes pharmaceutiques", 
                fontsize=22, fontweight='bold', pad=25)
    ax.set_xlabel("UMAP Dimension 1", fontsize=16, fontweight='bold')
    ax.set_ylabel("UMAP Dimension 2", fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.7)
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11, 
             frameon=True, shadow=True, fancybox=True, ncol=1)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Sauvegardé: {filename}")

def create_enhanced_umap_3d(embeddings, labels, le, filename="umap_3d_enhanced.png"):
    """Visualisation UMAP 3D matplotlib statique"""
    X = embeddings.astype(np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    
    N = X.shape[0]
    n_neighbors = max(15, min(100, int(np.sqrt(N))))
    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=0.1,
                       metric='cosine', random_state=42, n_jobs=1)
    z3 = reducer.fit_transform(X)
    
    fig = plt.figure(figsize=(20, 14), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='whitesmoke')
    
    # Utiliser le mapping de couleurs cohérent
    label_to_color = get_color_mapping(labels, le)
    unique_labels = sorted(np.unique(labels))
    
    for lab in unique_labels:
        mask = labels == lab
        if mask.sum() == 0:
            continue
        
        xi, yi, zi = z3[mask, 0], z3[mask, 1], z3[mask, 2]
        lab_name = str(le.inverse_transform([lab])[0])
        color = label_to_color[lab_name]
        
        ax.scatter(xi, yi, zi, s=70, alpha=0.85, marker='o',
                  edgecolor='black', linewidth=0.5, color=color,
                  label=lab_name, depthshade=True)
    
    ax.set_title("UMAP 3D - Visualisation spatiale des embeddings", 
                fontsize=22, fontweight='bold', pad=30)
    ax.set_xlabel("UMAP-1", fontsize=14, fontweight='bold', labelpad=12)
    ax.set_ylabel("UMAP-2", fontsize=14, fontweight='bold', labelpad=12)
    ax.set_zlabel("UMAP-3", fontsize=14, fontweight='bold', labelpad=12)
    
    ax.view_init(elev=20, azim=-65)
    ax.grid(True, alpha=0.2)
    
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10,
             frameon=True, shadow=True, ncol=1)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Sauvegardé: {filename}")
    
    return z3  # Retourner pour version interactive

def create_interactive_umap_3d(embeddings, labels, le, filename="umap_3d_interactive.html"):
    """Visualisation UMAP 3D interactive en HTML avec Plotly"""
    X = embeddings.astype(np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    
    N = X.shape[0]
    n_neighbors = max(15, min(100, int(np.sqrt(N))))
    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=0.1,
                       metric='cosine', random_state=42, n_jobs=1)
    z3 = reducer.fit_transform(X)
    
    # Utiliser le mapping de couleurs cohérent
    label_to_color = get_color_mapping(labels, le)
    
    # Créer DataFrame pour Plotly
    class_names = [str(le.inverse_transform([lab])[0]) for lab in labels]
    unique_labels_sorted = sorted(np.unique(labels))
    
    df_plot = pd.DataFrame({
        'UMAP-1': z3[:, 0],
        'UMAP-2': z3[:, 1],
        'UMAP-3': z3[:, 2],
        'Classe': class_names,
        'Label_num': labels
    })
    
    # Créer figure interactive avec couleurs personnalisées
    fig = go.Figure()
    
    for lab in unique_labels_sorted:
        lab_name = str(le.inverse_transform([lab])[0])
        mask = df_plot['Classe'] == lab_name
        df_subset = df_plot[mask]
        
        fig.add_trace(go.Scatter3d(
            x=df_subset['UMAP-1'],
            y=df_subset['UMAP-2'],
            z=df_subset['UMAP-3'],
            mode='markers',
            name=lab_name,
            marker=dict(
                size=4,
                color=label_to_color[lab_name],
                opacity=0.85,
                line=dict(width=0.3, color='black')  # Contour noir fin
            ),
            hovertemplate='<b>%{text}</b><br>UMAP-1: %{x:.2f}<br>UMAP-2: %{y:.2f}<br>UMAP-3: %{z:.2f}<extra></extra>',
            text=[lab_name] * len(df_subset)
        ))
    
    fig.update_layout(
        title='UMAP 3D Interactif - Exploration des embeddings pharmaceutiques',
        scene=dict(
            xaxis=dict(title='UMAP Dimension 1', backgroundcolor="rgb(240, 240, 240)", 
                      gridcolor="white", showbackground=True),
            yaxis=dict(title='UMAP Dimension 2', backgroundcolor="rgb(240, 240, 240)", 
                      gridcolor="white", showbackground=True),
            zaxis=dict(title='UMAP Dimension 3', backgroundcolor="rgb(240, 240, 240)", 
                      gridcolor="white", showbackground=True),
        ),
        font=dict(size=12),
        title_font_size=18,
        legend=dict(
            yanchor="top", 
            y=0.99, 
            xanchor="left", 
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=1
        ),
        width=1500,
        height=1000
    )
    
    fig.write_html(filename)
    print(f"✅ Sauvegardé: {filename} (ouvrir dans un navigateur)")

def create_tsne_visualizations(embeddings, labels, le):
    """Visualisations t-SNE 2D et 3D"""
    X = embeddings.astype(np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    
    # Utiliser le mapping de couleurs cohérent
    label_to_color = get_color_mapping(labels, le)
    unique_labels = sorted(np.unique(labels))
    
    # t-SNE 2D
    print("🔄 Calcul t-SNE 2D...")
    tsne_2d = TSNE(n_components=2, perplexity=50, max_iter=1000, 
                   random_state=42, n_jobs=-1)
    z2 = tsne_2d.fit_transform(X)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 14), facecolor='white')
    
    for lab in unique_labels:
        mask = labels == lab
        if mask.sum() == 0:
            continue
        
        xi, yi = z2[mask, 0], z2[mask, 1]
        lab_name = str(le.inverse_transform([lab])[0])
        color = label_to_color[lab_name]
        
        ax.scatter(xi, yi, s=80, alpha=0.8, marker='o',
                  edgecolor='black', linewidth=0.5, color=color,
                  label=lab_name, zorder=3)
    
    ax.set_title("t-SNE 2D - Visualisation alternative des embeddings", 
                fontsize=22, fontweight='bold', pad=25)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=16, fontweight='bold')
    ax.set_ylabel("t-SNE Dimension 2", fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.7)
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11,
             frameon=True, shadow=True, fancybox=True, ncol=1)
    
    plt.tight_layout()
    plt.savefig("tsne_2d.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Sauvegardé: tsne_2d.png")
    
    # t-SNE 3D
    print("🔄 Calcul t-SNE 3D...")
    tsne_3d = TSNE(n_components=3, perplexity=50, max_iter=1000,
                   random_state=42, n_jobs=-1)
    z3 = tsne_3d.fit_transform(X)
    
    # Version statique
    fig = plt.figure(figsize=(20, 14), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='whitesmoke')
    
    for lab in unique_labels:
        mask = labels == lab
        if mask.sum() == 0:
            continue
        
        xi, yi, zi = z3[mask, 0], z3[mask, 1], z3[mask, 2]
        lab_name = str(le.inverse_transform([lab])[0])
        color = label_to_color[lab_name]
        
        ax.scatter(xi, yi, zi, s=70, alpha=0.85, marker='o',
                  edgecolor='black', linewidth=0.5, color=color,
                  label=lab_name, depthshade=True)
    
    ax.set_title("t-SNE 3D - Représentation spatiale des classes", 
                fontsize=22, fontweight='bold', pad=30)
    ax.set_xlabel("t-SNE-1", fontsize=14, fontweight='bold', labelpad=12)
    ax.set_ylabel("t-SNE-2", fontsize=14, fontweight='bold', labelpad=12)
    ax.set_zlabel("t-SNE-3", fontsize=14, fontweight='bold', labelpad=12)
    
    ax.view_init(elev=20, azim=-65)
    ax.grid(True, alpha=0.2)
    
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10,
             frameon=True, shadow=True, ncol=1)
    
    plt.tight_layout()
    plt.savefig("tsne_3d.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Sauvegardé: tsne_3d.png")
    
    # Version interactive HTML - utilise le même mapping de couleurs
    class_names = [str(le.inverse_transform([lab])[0]) for lab in labels]
    
    df_plot = pd.DataFrame({
        't-SNE-1': z3[:, 0],
        't-SNE-2': z3[:, 1],
        't-SNE-3': z3[:, 2],
        'Classe': class_names
    })
    
    fig = go.Figure()
    
    for lab in unique_labels:
        lab_name = str(le.inverse_transform([lab])[0])
        mask = df_plot['Classe'] == lab_name
        df_subset = df_plot[mask]
        
        fig.add_trace(go.Scatter3d(
            x=df_subset['t-SNE-1'],
            y=df_subset['t-SNE-2'],
            z=df_subset['t-SNE-3'],
            mode='markers',
            name=lab_name,
            marker=dict(
                size=4,
                color=label_to_color[lab_name],
                opacity=0.85,
                line=dict(width=0.3, color='black')  # Contour noir fin
            ),
            hovertemplate='<b>%{text}</b><br>t-SNE-1: %{x:.2f}<br>t-SNE-2: %{y:.2f}<br>t-SNE-3: %{z:.2f}<extra></extra>',
            text=[lab_name] * len(df_subset)
        ))
    
    fig.update_layout(
        title='t-SNE 3D Interactif - Navigation dans l\'espace latent',
        scene=dict(
            xaxis=dict(title='t-SNE Dimension 1', backgroundcolor="rgb(240, 240, 240)", 
                      gridcolor="white", showbackground=True),
            yaxis=dict(title='t-SNE Dimension 2', backgroundcolor="rgb(240, 240, 240)", 
                      gridcolor="white", showbackground=True),
            zaxis=dict(title='t-SNE Dimension 3', backgroundcolor="rgb(240, 240, 240)", 
                      gridcolor="white", showbackground=True),
        ),
        font=dict(size=12),
        title_font_size=18,
        legend=dict(
            yanchor="top", 
            y=0.99, 
            xanchor="left", 
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=1
        ),
        width=1500,
        height=1000
    )
    
    fig.write_html("tsne_3d_interactive.html")
    print("✅ Sauvegardé: tsne_3d_interactive.html")

def create_condensed_calibration(probs_all, labels_all, le, num_classes):
    """Calibration condensée sur une seule figure"""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15), facecolor='white')
    axes = axes.flatten()
    
    brier_scores = []
    
    for c in range(num_classes):
        ax = axes[c]
        y_prob = probs_all[:, c]
        y_true = (labels_all == c).astype(int)
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=10, strategy='uniform')
        
        ax.plot(mean_predicted_value, fraction_of_positives, "o-", 
               linewidth=2.5, markersize=8, color='steelblue',
               label=f"{le.inverse_transform([c])[0]}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=2, alpha=0.6, 
               label="Calibration parfaite")
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlabel("Probabilité prédite", fontsize=10, fontweight='bold')
        ax.set_ylabel("Fraction observée", fontsize=10, fontweight='bold')
        
        brier = brier_score_loss(y_true, y_prob)
        brier_scores.append((le.inverse_transform([c])[0], brier))
        
        ax.set_title(f"{le.inverse_transform([c])[0]}\nBrier: {brier:.4f}", 
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
    
    # Supprimer le subplot vide (12ème)
    fig.delaxes(axes[-1])
    
    fig.suptitle("Courbes de calibration par classe pharmaceutique", 
                fontsize=20, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig("calibration_condensed.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Sauvegardé: calibration_condensed.png")
    
    # Graphique des Brier scores
    brier_df = pd.DataFrame(brier_scores, columns=['Classe', 'Brier'])
    brier_df = brier_df.sort_values('Brier')
    
    plt.figure(figsize=(14, 8), facecolor='white')
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(brier_df)))
    bars = plt.barh(brier_df['Classe'], brier_df['Brier'], color=colors, 
                    edgecolor='black', linewidth=1.5)
    plt.xlabel('Brier Score (plus bas = meilleur)', fontsize=14, fontweight='bold')
    plt.ylabel('Classe', fontsize=14, fontweight='bold')
    plt.title('Qualité de calibration par classe (Brier Score)', 
             fontsize=18, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig("brier_scores_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Sauvegardé: brier_scores_comparison.png")

def create_additional_visualizations(probs_all, labels_all, preds_all, le):
    """Graphiques supplémentaires avancés"""
    
    # 1. Distribution des probabilités maximales par classe
    max_probs = probs_all.max(axis=1)
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 12), facecolor='white')
    axes = axes.flatten()
    
    unique_labels = np.unique(labels_all)
    for idx, lab in enumerate(sorted(unique_labels)):
        ax = axes[idx]
        mask = labels_all == lab
        class_probs = max_probs[mask]
        
        ax.hist(class_probs, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(class_probs.mean(), color='red', linestyle='--', 
                  linewidth=2, label=f'Moyenne: {class_probs.mean():.3f}')
        ax.set_xlabel('Probabilité max', fontsize=10, fontweight='bold')
        ax.set_ylabel('Fréquence', fontsize=10, fontweight='bold')
        ax.set_title(f"{le.inverse_transform([lab])[0]}", 
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    fig.delaxes(axes[-1])
    fig.suptitle("Distribution des confiances de prédiction par classe", 
                fontsize=20, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig("confidence_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Sauvegardé: confidence_distribution.png")
    
    # 2. Matrice de confusion normalisée (%)
    conf_matrix = confusion_matrix(labels_all, preds_all)
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(16, 14), facecolor='white')
    class_names = [le.inverse_transform([lab])[0] for lab in sorted(unique_labels)]
    
    sns.heatmap(conf_matrix_norm, annot=True, fmt='.1f', cmap='YlGnBu',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Pourcentage (%)'}, vmin=0, vmax=100,
                linewidths=0.5, linecolor='gray')
    
    plt.title('Matrice de confusion normalisée (%)', 
             fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Classe prédite', fontsize=14, fontweight='bold')
    plt.ylabel('Classe réelle', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix_normalized.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Sauvegardé: confusion_matrix_normalized.png")
    
    # 3. Heatmap des probabilités moyennes
    prob_matrix = np.zeros((len(unique_labels), len(unique_labels)))
    for true_lab in unique_labels:
        for pred_lab in unique_labels:
            mask = labels_all == true_lab
            if mask.sum() > 0:
                prob_matrix[true_lab, pred_lab] = probs_all[mask, pred_lab].mean()
    
    plt.figure(figsize=(16, 14), facecolor='white')
    sns.heatmap(prob_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Probabilité moyenne'}, vmin=0, vmax=1,
                linewidths=0.5, linecolor='gray')
    
    plt.title('Heatmap des probabilités moyennes prédites', 
             fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Classe prédite', fontsize=14, fontweight='bold')
    plt.ylabel('Classe réelle', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("probability_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Sauvegardé: probability_heatmap.png")

# ==================== MAIN ====================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device}")
    
    model, tokenizer, training_state = load_model(num_classes, device)
    
    if model is None:
        print("\n🚀 Aucun modèle trouvé, lancement de l'entraînement...\n")
        
        tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        train_encodings = tokenizer(train_df['text'].tolist(), truncation=True,
                                    padding=True, max_length=64)
        val_encodings = tokenizer(val_df['text'].tolist(), truncation=True,
                                  padding=True, max_length=64)
        
        train_labels = train_df['label_encoded'].values
        val_labels = val_df['label_encoded'].values
        
        train_dataset = PharmaDataset(train_encodings, train_labels)
        val_dataset = PharmaDataset(val_encodings, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=16, num_workers=0)
        
        model = AutoModelForSequenceClassification.from_pretrained(
            "camembert-base", num_labels=num_classes)
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=2e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=1)
        
        epochs = 3
        train_losses, val_losses = [], []
        val_accuracies, val_f1_scores, train_f1_scores = [], [], []
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            all_train_preds, all_train_labels = [], []
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Train"):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    preds = torch.argmax(outputs.logits, dim=1)
                    all_train_preds.extend(preds.cpu().numpy())
                    all_train_labels.extend(labels.cpu().numpy())
            
            epoch_train_loss = total_loss/len(train_loader)
            train_losses.append(epoch_train_loss)
            train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')
            train_f1_scores.append(train_f1)
            
            model.eval()
            all_val_preds, all_val_labels = [], []
            total_val_loss = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Val"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    total_val_loss += outputs.loss.item()
                    
                    preds = torch.argmax(outputs.logits, dim=1)
                    all_val_preds.extend(preds.cpu().numpy())
                    all_val_labels.extend(labels.cpu().numpy())
            
            epoch_val_loss = total_val_loss/len(val_loader)
            val_losses.append(epoch_val_loss)
            acc = accuracy_score(all_val_labels, all_val_preds)
            f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
            val_accuracies.append(acc)
            val_f1_scores.append(f1)
            
            print(f"Epoch {epoch+1}: Train Loss={epoch_train_loss:.4f}, "
                  f"Val Loss={epoch_val_loss:.4f}, Val Acc={acc:.4f}, Val F1={f1:.4f}")
            scheduler.step(f1)
        
        metrics = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'val_f1_scores': val_f1_scores,
            'train_f1_scores': train_f1_scores
        }
        save_model(model, tokenizer, optimizer, epochs, metrics)
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs+1), train_losses, 'o-', color='blue', label='Train')
        plt.plot(range(1, epochs+1), val_losses, 'o-', color='red', label='Validation')
        plt.title('Évolution des pertes', fontweight='bold')
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.xticks(range(1, epochs+1))
        plt.grid(True, alpha=0.3); plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs+1), train_f1_scores, 'o-', color='orange', label='Train')
        plt.plot(range(1, epochs+1), val_f1_scores, 'o-', color='green', label='Validation')
        plt.title('Évolution des F1-scores', fontweight='bold')
        plt.xlabel('Epoch'); plt.ylabel('F1-score')
        plt.xticks(range(1, epochs+1)); plt.ylim(0, 1)
        plt.grid(True, alpha=0.3); plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_metrics.png', dpi=300)
        plt.show()
    
    else:
        print("⏭️  Utilisation du modèle pré-entraîné")
        val_encodings = tokenizer(val_df['text'].tolist(), truncation=True,
                                  padding=True, max_length=64)
        val_labels = val_df['label_encoded'].values
        val_dataset = PharmaDataset(val_encodings, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=16, num_workers=0)
    
    # ==================== ÉVALUATIONS ====================
    print("\n📊 Évaluation du modèle...")
    
    model.eval()
    preds, true_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Prédictions finales"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_preds = torch.argmax(outputs.logits, dim=1)
            preds.extend(batch_preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    preds_classes = le.inverse_transform(preds)
    true_classes = le.inverse_transform(true_labels)
    
    print("\n" + "="*50)
    print("Rapport de classification:")
    print("="*50)
    print(classification_report(true_classes, preds_classes, digits=4, zero_division=0))
    
    conf_matrix = confusion_matrix(true_classes, preds_classes)
    unique_classes = sorted(set(true_classes) | set(preds_classes))
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=unique_classes, yticklabels=unique_classes,
                cbar_kws={'label': 'Nombre de prédictions'}, linewidths=0.5)
    plt.title('Matrice de Confusion - Classification Pharmaceutique', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Prédictions', fontsize=13, fontweight='bold')
    plt.ylabel('Vraies étiquettes', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()
    
    # ==================== VISUALISATIONS AVANCÉES ====================
    print("\n🎨 Génération des visualisations avancées...")
    
    probs_all, labels_all, preds_all, embeddings_all = [], [], [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecte embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds_batch = np.argmax(probs, axis=1)
            
            last_hidden = outputs.hidden_states[-1].cpu().numpy()
            cls_embeddings = last_hidden[:, 0, :]
            
            probs_all.append(probs)
            labels_all.append(labels.cpu().numpy())
            preds_all.append(preds_batch)
            embeddings_all.append(cls_embeddings)
    
    probs_all = np.vstack(probs_all)
    labels_all = np.concatenate(labels_all)
    preds_all = np.concatenate(preds_all)
    embeddings_all = np.vstack(embeddings_all)
    
    # UMAP visualizations
    print("\n🔵 Génération UMAP 2D...")
    create_enhanced_umap_2d(embeddings_all, labels_all, le)
    
    print("\n🔵 Génération UMAP 3D statique...")
    create_enhanced_umap_3d(embeddings_all, labels_all, le)
    
    print("\n🌐 Génération UMAP 3D interactif (HTML)...")
    create_interactive_umap_3d(embeddings_all, labels_all, le)
    
    # t-SNE visualizations
    print("\n🟠 Génération visualisations t-SNE...")
    create_tsne_visualizations(embeddings_all, labels_all, le)
    
    # Calibration condensée
    print("\n📈 Génération courbes de calibration...")
    create_condensed_calibration(probs_all, labels_all, le, num_classes)
    
    # Graphiques supplémentaires
    print("\n📊 Génération graphiques supplémentaires...")
    create_additional_visualizations(probs_all, labels_all, preds_all, le)
    
    # Confiance vs Accuracy
    probs_max = probs_all.max(axis=1)
    sorted_idx = np.argsort(-probs_max)
    sorted_preds = preds_all[sorted_idx]
    sorted_labels = labels_all[sorted_idx]
    sorted_conf = probs_max[sorted_idx]
    
    thresholds = np.linspace(0.5, 0.99, 20)
    coverage, accuracy_at_thresh = [], []
    for t in thresholds:
        keep = sorted_conf >= t
        if keep.sum() == 0:
            coverage.append(0.0)
            accuracy_at_thresh.append(np.nan)
        else:
            coverage.append(keep.mean())
            accuracy_at_thresh.append((sorted_preds[keep] == sorted_labels[keep]).mean())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')
    
    ax1.plot(thresholds, accuracy_at_thresh, marker='o', linewidth=3, 
            markersize=8, color='darkgreen', label='Accuracy')
    ax1.set_xlabel('Seuil minimal de confiance', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Performance vs Confiance du modèle', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3); ax1.set_ylim(0, 1)
    ax1.legend(fontsize=11)
    
    ax2.plot(thresholds, coverage, marker='s', linewidth=3, 
            markersize=8, color='darkblue', label='Couverture')
    ax2.set_xlabel('Seuil minimal de confiance', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Proportion de données conservées', fontsize=12, fontweight='bold')
    ax2.set_title('Couverture vs Seuil de confiance', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3); ax2.set_ylim(0, 1)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig("accuracy_vs_confidence.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Sauvegardé: accuracy_vs_confidence.png")
    
    print("\n✅ Toutes les visualisations ont été générées!")
    
    # ==================== PRÉDICTIONS NOUVELLES DONNÉES ====================
    print("\n🔮 Prédiction sur nouvelles données...")
    X_new['text'] = X_new['Avis.Pharmaceutique'].apply(preprocess_text)
    new_encodings = tokenizer(X_new['text'].tolist(), truncation=True,
                             padding=True, max_length=64)
    
    class PredictionDataset(Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        def __len__(self):
            return len(self.encodings['input_ids'])
        def __getitem__(self, idx):
            return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
    
    new_dataset = PredictionDataset(new_encodings)
    new_loader = DataLoader(new_dataset, batch_size=16, shuffle=False)
    
    model.eval()
    preds_new = []
    
    with torch.no_grad():
        for batch in tqdm(new_loader, desc="Prédictions nouvelles données"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_preds = torch.argmax(outputs.logits, dim=1)
            preds_new.extend(batch_preds.cpu().numpy())
    
    preds_new_classes = le.inverse_transform(preds_new)
    df_export = pd.DataFrame({
        "Colonne1": df_new.iloc[:, 0],
        "pred": preds_new_classes
    })
    df_export.to_csv("predictions.csv", index=False)
    print("✅ Prédictions exportées dans predictions.csv")
    
    print("\n" + "="*60)
    print("🎉 SCRIPT TERMINÉ AVEC SUCCÈS!")
    print("="*60)
    print("\n📁 Fichiers générés:")
    print("   • umap_2d_enhanced.png")
    print("   • umap_3d_enhanced.png")
    print("   • umap_3d_interactive.html ⭐ (à ouvrir dans le navigateur)")
    print("   • tsne_2d.png")
    print("   • tsne_3d.png")
    print("   • tsne_3d_interactive.html ⭐ (à ouvrir dans le navigateur)")
    print("   • calibration_condensed.png")
    print("   • brier_scores_comparison.png")
    print("   • confidence_distribution.png")
    print("   • confusion_matrix_normalized.png")
    print("   • probability_heatmap.png")
    print("   • accuracy_vs_confidence.png")
    print("   • confusion_matrix.png")
    print("   • predictions.csv")
    print("="*60)