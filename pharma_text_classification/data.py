import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Lire le fichier CSV
data = pd.read_csv(r'data_defi2.csv', sep=';')
# df_new = pd.read_csv(r'.csv', sep=';')

# X_new = df_new.copy()

# Aperçu général
print(data.head(), "\n")

print(data.info(), "\n")

print(data.describe(include='object'), "\n")

# Conversion en classe principale
print(data['PLT'].unique())
data['PLT'] = pd.to_numeric(data['PLT'], errors='coerce').fillna(0).astype(int)
print(data['PLT'].unique())

# Vérification des valeurs manquantes
print(data.isna().sum(), "\n")

# On choisit de supprimer ces lignes (373)
data = data.dropna(subset=['Avis.Pharmaceutique'])


# Longueur des commentaires
text_col = 'Avis.Pharmaceutique' 
if text_col in data.columns:
    data['text_length'] = data[text_col].astype(str).apply(lambda x: len(x.split()))
    plt.figure(figsize=(8, 4))
    sns.histplot(data['text_length'], bins=30, kde=True)
    plt.title("Distribution de la longueur des commentaires (en mots)")
    plt.xlabel("Nombre de mots")
    plt.ylabel("Fréquence")
    plt.show()

    print(f"Longueur moyenne des textes : {data['text_length'].mean():.2f} mots")
    print(f"Longueur min : {data['text_length'].min()} / max : {data['text_length'].max()}\n")


# Extraire la colonne texte
texts = data["Libellé.Prescription"].dropna().astype(str)
# Fonction pour extraire les abréviations potentielles
def extract_abbr(text):
    # On garde les tokens en majuscules ou mixtes (ex: "CPR", "IV", "NaCl")
    tokens = re.findall(r"\b[A-Z]{2,}\b", text)
    return tokens
# Appliquer sur toutes les lignes
all_tokens = []
for t in texts:
    all_tokens.extend(extract_abbr(t))

# Compter les occurrences
counter = Counter(all_tokens)

# Mettre dans un DataFrame pour lecture plus claire
df_abbr = pd.DataFrame(counter.most_common(30), columns=["Abréviation", "Fréquence"])
print(df_abbr)


def preprocess_text(text):
    text = str(text).lower()               # uniformiser en minuscules
    text = text.replace('\n', ' ')         # supprimer les sauts de ligne
    text = text.replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)       # normaliser les espaces
    text = re.sub(r'[^\w\s%/.=<>-]', '', text)  # garder ponctuation utile médicale
    return text.strip()

data['text'] = data['Avis.Pharmaceutique'].apply(preprocess_text)

# Encodage des labels
le = LabelEncoder()
data['label_encoded'] = le.fit_transform(data['PLT'])
num_classes = len(le.classes_)
# Split train / validation
train_df, val_df = train_test_split(
    data,
    test_size=0.1,
    stratify=data['label_encoded'],
    random_state=42
)

