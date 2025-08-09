# 🎓 UIC RAG Chatbot (FR/AR) — Single Page

Chatbot bilingue (**Français** / **Arabe**) pour l’Université Internationale de Casablanca (**UIC**), utilisant une architecture **RAG (Retrieval-Augmented Generation)** avec **Google Gemini**.

- **Interface unique** : une seule page web (`templates/chatbot.html`) avec style personnalisable (`static/chatbot.css`)
- **Bilingue automatique** : réponse dans la langue de la question (FR ou AR)
- **Recherche sémantique** : index vectoriel FAISS construit à partir des Q/R (`data/data_uic.csv` ou `.xlsx`)
- **Réponses précises** : Gemini ne répond qu’à partir du contexte fourni, sinon message de fallback

## 📂 Structure du projet

```
uic_rag/
├─ app_rag.py               # API Flask - sert l'UI + endpoint /predict
├─ rag/
│  ├─ ingest.py              # Création de l'index vectoriel (FAISS)
│  └─ rag_pipeline.py        # Recherche + prompt + génération Gemini
├─ data/
│  └─ data_uic.csv           # Dataset bilingue (Pattern, Tag, response)
├─ vecstore/                 # Index générés (index.faiss + meta.pkl)
├─ templates/
│  └─ chatbot.html           # UI du chatbot
├─ static/
│  └─ chatbot.css            # Styles (support RTL)
├─ .env                      # Clé API Gemini (non versionné)
├─ .gitignore                # Fichiers/dirs ignorés par Git
├─ requirements.txt          # Dépendances Python
├─ Dockerfile                # Build image Docker
├─ docker-compose.yml        # Lancement ingestion + app
└─ README.md                 # Ce fichier
```

## ⚙️ Installation locale (sans Docker)

### 1. Prérequis
- Python 3.10+
- Clé API **Google Gemini** : [console](https://makersuite.google.com/app/apikey)

### 2. Installation
```bash
git clone https://github.com/<votre_user>/uic_rag.git
cd uic_rag

python -m venv .venv
# Windows :
.venv\Scripts\activate
# Mac/Linux :
source .venv/bin/activate

pip install -r requirements.txt
cp .env.sample .env
```

**Remplir `.env`** :
```
GEMINI_API_KEY=VOTRE_CLE_ICI
GEMINI_MODEL=gemini-1.5-flash
EMBED_MODEL=text-embedding-004
```

### 3. Construire l’index vectoriel
```bash
python rag/ingest.py
```

### 4. Lancer l’application
```bash
python app_rag.py
```
Ouvrez [http://localhost:8000](http://localhost:8000)

## 🐳 Utilisation avec Docker

### 1. Build & run
```bash
docker compose up --build
```
Puis [http://localhost:8000](http://localhost:8000)

### 2. Mise à jour des données
- Modifier `data/data_uic.csv`
- Relancer ingestion :
```bash
docker compose run --rm ingest
docker compose restart uic_rag
```

## 🔄 Flux de fonctionnement

1. **Ingestion (`ingest.py`)**
   - Lit `data/data_uic.csv`
   - Crée des embeddings avec `text-embedding-004`
   - Stocke dans un index **FAISS** + métadonnées

2. **Requête utilisateur (`/predict`)**
   - L’UI envoie la question (FR ou AR) en JSON
   - `rag_pipeline.py` cherche les entrées proches dans l’index
   - Construit un **prompt** avec contexte uniquement
   - Appelle **Gemini** (`gemini-1.5-flash` ou autre)
   - Renvoie la réponse (dans la langue détectée)

## 🗂 Format du dataset

`data/data_uic.csv` ou `.xlsx` doit contenir :

| Pattern                                | Tag          | response                                       |
|----------------------------------------|--------------|------------------------------------------------|
| Comment s'inscrire à l'UIC ?           | Admissions   | Pour vous inscrire à l'UIC, ...                |
| كيف أسجّل في جامعة UIC؟                | Admissions   | للتسجيل في UIC، أنشئ حسابًا ...                 |

- **Pattern** : Question ou formulation possible (FR ou AR)
- **Tag** : Catégorie (Admissions, Frais, Programmes…)
- **response** : Réponse exacte
