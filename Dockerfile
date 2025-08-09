FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dépendances système minimales (certifi + locales)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates locales \
 && rm -rf /var/lib/apt/lists/* \
 && update-ca-certificates

# (facultatif) locale fr
RUN sed -i 's/# fr_FR.UTF-8 UTF-8/fr_FR.UTF-8 UTF-8/' /etc/locale.gen && locale-gen

WORKDIR /app

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code
COPY . .

# User non-root
RUN useradd -m appuser
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Lancement par défaut : gunicorn (ingestion gérée par docker-compose)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "app_rag:app"]
