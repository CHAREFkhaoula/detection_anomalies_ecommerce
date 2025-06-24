import os
import time
import json
import pandas as pd
from kafka import KafkaProducer
from datetime import datetime

def load_transactions_data(csv_path):
    """Charge les données de transactions depuis un fichier CSV"""
    df = pd.read_csv(csv_path, sep=',')
    df = df.sort_values(by="timestamp")  # Tri chronologique
    return df

def create_kafka_producer(broker_url):
    """Crée et retourne un producteur Kafka configuré"""
    return KafkaProducer(
        bootstrap_servers=[broker_url],
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        api_version=(2, 8, 1),  # Version API explicite
        request_timeout_ms=30000,  # Timeout augmenté
        retries=5,  # Plus de tentatives
        reconnect_backoff_ms=1000  # Délai entre reconnexions
    )


def prepare_transaction(row):
    """Prépare un dictionnaire avec les champs pertinents de la transaction"""
    return row[[
        "transaction_id", "user_id", "timestamp", "amount", "quantity", "hour_of_day",
        "day_of_week", "month", "time_since_last", "ip_geodist", "amount_per_item",
        "user_segment", "payment_method", "country", "rolling_mean_amount_5",
        "rolling_std_amount_5", "rolling_max_amount_5", "rolling_mean_amount_10",
        "rolling_std_amount_10", "rolling_max_amount_10", "rolling_mean_amount_30",
        "rolling_std_amount_30", "rolling_max_amount_30", "amount_deviation_5",
        "amount_deviation_10", "amount_deviation_30"
    ]].to_dict()

def simulate_real_time_stream(producer, df, topic, speedup=60):
    """Envoie les transactions en simulant un flux temps réel"""
    previous_ts = None
    
    for _, row in df.iterrows():
        current_ts = pd.to_datetime(row["timestamp"])
        
        # Calcul du délai entre transactions
        if previous_ts is not None:
            diff_seconds = (current_ts - previous_ts).total_seconds()
            time.sleep(diff_seconds / speedup)
        
        transaction = prepare_transaction(row)
        
        try:
            producer.send(topic, transaction)
            print(f"Envoyé: ID {transaction['transaction_id']} | {transaction['timestamp']}")
            previous_ts = current_ts
        except Exception as e:
            print(f"Erreur lors de l'envoi: {str(e)}")
            time.sleep(5)  # Pause avant réessai

def wait_for_kafka(broker_url, max_retries=5, delay=30):
    """Attend que Kafka soit disponible"""
    from kafka import KafkaAdminClient
    from kafka.errors import NoBrokersAvailable
    
    """Attend que Kafka soit disponible"""
    for i in range(max_retries):
        try:
            # Test de connexion simplifié
            producer = KafkaProducer(
                bootstrap_servers=[broker_url],
                api_version=(2, 8, 1),  # Force la version API
                request_timeout_ms=20000
            )
            producer.close()
            print("Connecté à Kafka avec succès!")
            return True
        except Exception as e:
            print(f"Attente de Kafka ({i+1}/{max_retries}) - Erreur: {str(e)}")
            time.sleep(delay)
    raise Exception("Impossible de se connecter à Kafka après plusieurs tentatives")

if __name__ == "__main__":
    # Configuration
    CSV_FILE = "large_ecommerce_transactions.csv"
    # KAFKA_BROKER = os.environ.get('KAFKA_BROKER', 'kafka:9092')
    # KAFKA_BROKER = 'localhost:9092'
    KAFKA_BROKER = "kafka:29092" if os.getenv('IN_DOCKER') else "localhost:9092"
    TOPIC_NAME = 'ecommerce'
    SPEEDUP_FACTOR = 60  # Accélération de la simulation
    
    try:
        # 1. Chargement des données
        print("Chargement des données de transactions...")
        df = load_transactions_data(CSV_FILE)
        
        # 2. Attente que Kafka soit disponible
        wait_for_kafka(KAFKA_BROKER)
        
        # 3. Création du producteur
        producer = create_kafka_producer(KAFKA_BROKER)
        
        # 4. Simulation du flux temps réel
        print("Début de l'envoi des transactions...")
        simulate_real_time_stream(producer, df, TOPIC_NAME, SPEEDUP_FACTOR)
        
        # 5. Finalisation
        producer.flush()
        print("Toutes les transactions ont été envoyées avec succès!")
        
    except Exception as e:
        print(f"Erreur dans le producteur: {str(e)}")
    finally:
        if 'producer' in locals():
            producer.close()