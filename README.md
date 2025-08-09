# UIC RAG Chatbot (FR/AR) — Single Page
This repository hosts a single-page UI (templates/chatbot.html) backed by a Flask API using RAG + Gemini.

## Quick start
1) `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
2) `pip install -r requirements.txt`
3) `cp .env.sample .env` and set `GEMINI_API_KEY`
4) Put your knowledge base `data_uic.xlsx` into `data/`
5) Fill code in `ingest.py`, `rag_pipeline.py`, `app_rag.py` (will be provided next step)
6) Run `python ingest.py`
7) Run `python app_rag.py` and open http://localhost:8000/

## Deploy
- `gunicorn -w 2 -b 0.0.0.0:8000 app_rag:app`
- Or Docker: `docker compose up --build`
