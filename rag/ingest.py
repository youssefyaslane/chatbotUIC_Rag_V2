import os
import sys
import pickle
import math
from typing import List

import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
import google.generativeai as genai

# ---- Paths robustes (depuis le fichier courant) ----
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA_DIR = os.path.join(ROOT, "data")
VEC_DIR = os.path.join(ROOT, "vecstore")

DATA_FILE = os.path.join(DATA_DIR, "data_uic.csv")   # <-- ton fichier
INDEX_FILE = os.path.join(VEC_DIR, "index.faiss")
META_FILE = os.path.join(VEC_DIR, "meta.pkl")

def read_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    # CSV attendu (FR + AR) avec colonnes Pattern, Tag, response
    df = pd.read_csv(path)
    for col in ["Pattern", "Tag", "response"]:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante dans {path}: {col}")
    return df

def embed_batch(texts: List[str], model_name: str, batch_size: int = 32) -> np.ndarray:
    """Embeddings Gemini en batch (plus stable pour gros volumes)."""
    vecs = []
    n = len(texts)
    for i in range(0, n, batch_size):
        chunk = texts[i:i+batch_size]
        # L'API embed_content ne prend pas de liste => on boucle
        for t in chunk:
            emb = genai.embed_content(model=model_name, content=t)
            vecs.append(np.array(emb["embedding"], dtype="float32"))
        print(f"  - Embeddings: {min(i+batch_size, n)}/{n}")
    return np.vstack(vecs)

def main():
    # Charge .env à la racine
    load_dotenv(os.path.join(ROOT, ".env"))

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY manquante dans .env")
    genai.configure(api_key=api_key)

    embed_model = os.getenv("EMBED_MODEL", "text-embedding-004")

    print(f"🔎 Lecture du dataset: {DATA_FILE}")
    df = read_dataset(DATA_FILE)

    # Texte indexé = Pattern + Réponse (ça aide la recherche FR/AR)
    texts = (df["Pattern"].fillna("") + " || Réponse: " + df["response"].fillna("")).tolist()
    metas = df.to_dict(orient="records")

    print(f"🧠 Génération d'embeddings ({len(texts)} éléments) avec {embed_model} …")
    mat = embed_batch(texts, model_name=embed_model, batch_size=32)

    # Normalisation L2 pour utiliser IP comme cosinus
    faiss.normalize_L2(mat)
    dim = mat.shape[1]

    print(f"📦 Construction de l'index FAISS (dim={dim}) …")
    index = faiss.IndexFlatIP(dim)
    index.add(mat)

    os.makedirs(VEC_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump({"metas": metas, "texts": texts}, f)

    print(f"✅ Index écrit : {INDEX_FILE}")
    print(f"✅ Métadonnées : {META_FILE}")
    print("🎯 Ingestion terminée.")

if __name__ == "__main__":
    main()
