FROM bitnami/spark:3.5.1
# Créer un dossier de travail
WORKDIR /app
RUN pip install pandas
# Copier les requirements
COPY requirements.txt .

# Installer les dépendances Python
RUN pip3 install -r requirements.txt

# Copier tout le dossier app dans /app/app
COPY app ./app

# Commande par défaut si tu veux juste tester
CMD ["python3", "app/consumer_test.py"]