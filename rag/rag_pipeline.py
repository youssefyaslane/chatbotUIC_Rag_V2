import os
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
import google.generativeai as genai

# ---- Paths robustes ----
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
VEC_DIR = os.path.join(ROOT, "vecstore")
INDEX_FILE = os.path.join(VEC_DIR, "index.faiss")
META_FILE  = os.path.join(VEC_DIR, "meta.pkl")

class RAGPipeline:
    def __init__(self, score_min: float = 0.2):
        """
        score_min : seuil (cosinus approx via IP) pour filtrer le hors-sujet.
                    Ajuste si nécessaire (0.0 à 1.0). 0.2 est un point de départ.
        """
        # Charge .env à la racine
        load_dotenv(os.path.join(ROOT, ".env"))
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY manquante dans .env")
        genai.configure(api_key=api_key)

        self.gen_model   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.embed_model = os.getenv("EMBED_MODEL", "text-embedding-004")
        self.score_min   = float(score_min)

        if not os.path.exists(INDEX_FILE):
            raise FileNotFoundError("Index FAISS introuvable. Lance d’abord: python rag/ingest.py")
        self.index = faiss.read_index(INDEX_FILE)

        if not os.path.exists(META_FILE):
            raise FileNotFoundError("Métadonnées introuvables. Lance d’abord: python rag/ingest.py")
        
        with open(META_FILE, "rb") as f:
            store = pickle.load(f)
        self.metas = store["metas"]
        self.texts = store["texts"]

    def _embed(self, text: str) -> np.ndarray:
        v = genai.embed_content(model=self.embed_model, content=text)["embedding"]
        v = np.array(v, dtype="float32")[None, :]
        faiss.normalize_L2(v)
        return v

    def retrieve(self, query: str, k: int = 5):
        qv = self._embed(query)
        scores, idxs = self.index.search(qv, k)
        items = []
        for i, s in zip(idxs[0].tolist(), scores[0].tolist()):
            if i == -1:
                continue
            # Filtrage par score min pour limiter le hors-sujet
            if s < self.score_min:
                continue
            items.append({
                "score": float(s),
                "text": self.texts[i],
                "meta": self.metas[i],
            })
        return items

    def _build_prompt(self, query: str, retrieved):
        if retrieved:
            ctx_lines = []
            for it in retrieved:
                m = it["meta"]
                ctx_lines.append(
                    f"- TAG: {m.get('Tag','')}\n"
                    f"  Exemple/Question: {m.get('Pattern','')}\n"
                    f"  Réponse: {m.get('response','')}"
                )
            context_block = "\n".join(ctx_lines)
        else:
            context_block = "Aucun contexte pertinent trouvé."

        system_msg = (
            "أنت مستشار جامعة UIC / You are UIC assistant. "
            "Answer strictly from the provided context only. "
            "Reply in the SAME language as the user message (Arabic if Arabic, French if French). "
            "If the info is missing in context, reply politely:\n"
            "FR: \"Je n’ai pas cette information pour le moment.\" |\n"
            "AR: \"لا أتوفر على هذه المعلومة حاليًا.\""
        )

        user_msg = (
            f"Question:\n{query}\n\n"
            f"Contexte:\n{context_block}"
        )
        return system_msg, user_msg

# --- dans rag_pipeline.py ---

    def generate(self, query: str):
        retrieved = self.retrieve(query, k=5)
        sys, user = self._build_prompt(query, retrieved)

        # ✅ CORRECTIF: passer le prompt système via system_instruction=
        model = genai.GenerativeModel(
            model_name=self.gen_model,
            system_instruction=sys,        # <-- au lieu d'envoyer un "role: system"
        )

        # Et on n'envoie QUE le message utilisateur dans generate_content
        resp = model.generate_content(user)

        # Récupération robuste du texte
        text = ""
        try:
            text = (getattr(resp, "text", None) or "").strip()
            if not text and hasattr(resp, "candidates") and resp.candidates:
                parts = resp.candidates[0].content.parts
                if parts and hasattr(parts[0], "text"):
                    text = (parts[0].text or "").strip()
        except Exception:
            pass

        if not text:
            text = "Je n’ai pas cette information pour le moment."

        top_tag = retrieved[0]["meta"].get("Tag", "") if retrieved else ""
        return text, top_tag, retrieved
