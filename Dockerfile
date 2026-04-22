# Das offizielle Image für TF 2.10 mit GPU-Support
FROM tensorflow/tensorflow:2.10.0-gpu

WORKDIR /app

# Installiere System-Abhängigkeiten
RUN apt-get update && apt-get install -y \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Wir installieren deine Abhängigkeiten direkt via Pip
# Das entspricht exakt deiner .yml Liste
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    pandas \
    matplotlib \
    scipy \
    scikit-learn \
    networkx \
    seaborn \
    sqlalchemy \
    flask \
    alembic \
    pyyaml \
    tqdm \
    pillow \
    cloudpickle \
    bcrypt \
    spacy==3.7.5 \
    spacy-lookups-data \
    mlflow \
    cryptography

# Kopiere deinen Code in den Container
COPY . .

# Falls du ein spacy-Modell brauchst, lade es hier direkt herunter:
RUN python3 -m spacy download en_core_web_sm

CMD ["python3", "_hyperparameterTuning.py"]
