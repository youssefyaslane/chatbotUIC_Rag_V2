# app_rag.py
import os
from flask import Flask, render_template, request, jsonify
from rag.rag_pipeline import RAGPipeline

# Optionnel : activer CORS si tu sers le front depuis un autre domaine
# from flask_cors import CORS

def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    # CORS(app)  # décommente si besoin

    # Charger le pipeline RAG une seule fois
    app.rag = RAGPipeline()

    @app.route("/", methods=["GET"])
    def home():
        # Sert ta SEULE page
        return render_template("chatbot.html")

    @app.route("/health", methods=["GET"])
    def health():
        return {"status": "ok"}, 200

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            data = request.get_json(force=True) or {}
            text = (data.get("text") or "").strip()
            if not text:
                return jsonify({"reply": "Entrez un message / أدخل رسالة.", "tag": ""}), 200

            reply, tag, _ = app.rag.generate(text)
            return jsonify({"reply": reply, "tag": tag}), 200

        except Exception as e:
            # Log minimal en console; évite d'exposer des détails en prod
            print(f"[ERROR] /predict: {e}")
            return jsonify({"reply": "Erreur interne. Réessaie dans un instant.", "tag": ""}), 500

    return app


if __name__ == "__main__":
    # Permet de configurer host/port via variables d'env
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"

    app = create_app()
    app.run(host=host, port=port, debug=debug)
