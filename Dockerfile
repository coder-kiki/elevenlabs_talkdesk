# Basieren auf einem schlanken offiziellen Python-Image
FROM python:3.12-slim

# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

# Kopiere zuerst nur die requirements-Datei und installiere die Pakete
# (Nutzt Docker-Caching, sodass Pakete nicht bei jeder Code-Änderung neu installiert werden)
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere unser Haupt-Skript in den Container
COPY agent_server.py .

# Informiere Docker, dass die Anwendung im Container auf Port 8080 lauscht
# Wichtig für das Port-Mapping durch Elest.io CI/CD!
EXPOSE 8080

# Standardbefehl zum Starten der Anwendung, wenn der Container gestartet wird
# Umgebungsvariablen (API Keys etc.) werden von Elest.io CI/CD bereitgestellt
CMD ["python3", "agent_server.py"]