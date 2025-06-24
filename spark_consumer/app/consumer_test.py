from pyspark.sql import SparkSession 
from pyspark.sql.functions import col, from_json, struct, current_timestamp
from pyspark.sql.types import *
import json
import joblib
import numpy as np
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
import psycopg2  # PostgreSQL driver
from psycopg2 import sql

def get_db_connection():
    connection = psycopg2.connect(
        host="timescaledb",         # Docker service name dans le réseau Docker
        database="anomalie_detection",
        user="user",
        password="password",
        port=5432                   # pour être explicite
    )
    return connection

def get_timescale_connection():
    return {
        "url": "jdbc:postgresql://timescaledb:5432/anomalie_detection",
        "driver": "org.postgresql.Driver",
        "user": "user",
        "password": "password",
        "dbtable": "transactions"
    }

# === Chemins des artefacts ===
MODEL_PATHS = {
    'model': "/app/model/isolation_forest_model.joblib",
    'threshold': "/app/model/best_isolation_forest_threshold.json",
    'scaler': "/app/model/scaler.joblib",
    'payment_columns': "/app/model/payment_method_columns.json",
    'country_columns': "/app/model/country_columns.json",
    'segment_encoder': "/app/model/user_segment_encoder.joblib"
}

def load_artifacts():
    """Charge tous les artefacts nécessaires"""
    artifacts = {}
    artifacts['model'] = joblib.load(MODEL_PATHS['model'])
    artifacts['scaler'] = joblib.load(MODEL_PATHS['scaler'])
    artifacts['encoder'] = joblib.load(MODEL_PATHS['segment_encoder'])

    with open(MODEL_PATHS['threshold']) as f:
        artifacts['threshold'] = json.load(f).get('threshold', -0.2)
    with open(MODEL_PATHS['payment_columns']) as f:
        artifacts['payment_columns'] = json.load(f)
    with open(MODEL_PATHS['country_columns']) as f:
        artifacts['country_columns'] = json.load(f)
    return artifacts

def create_spark_session():
    return SparkSession.builder \
        .appName("EcommerceFraudCluster") \
        .master("spark://spark-master:7077") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.postgresql:postgresql:42.2.23") \
        .config("spark.executor.memory", "512m") \
        .config("spark.executor.cores", "1") \
        .config("spark.driver.memory", "512m") \
        .config("spark.executor.instances", "1") \
        .getOrCreate()

def main():
    print("Chargement des artefacts ML...")
    artifacts = load_artifacts()
    spark = create_spark_session()

    print("Broadcast des artefacts...")
    broadcast_vars = {
        'model': spark.sparkContext.broadcast(artifacts['model']),
        'scaler': spark.sparkContext.broadcast(artifacts['scaler']),
        'encoder': spark.sparkContext.broadcast(artifacts['encoder']),
        'threshold': spark.sparkContext.broadcast(artifacts['threshold']),
        'payment_columns': spark.sparkContext.broadcast(artifacts['payment_columns']),
        'country_columns': spark.sparkContext.broadcast(artifacts['country_columns'])
    }

    schema = StructType([
        StructField("transaction_id", StringType()),
        StructField("user_id", StringType()),
        StructField("timestamp", StringType()),
        StructField("amount", DoubleType()),
        StructField("quantity", IntegerType()),
        StructField("hour_of_day", IntegerType()),
        StructField("day_of_week", IntegerType()),
        StructField("month", IntegerType()),
        StructField("time_since_last", DoubleType()),
        StructField("ip_geodist", DoubleType()),
        StructField("amount_per_item", DoubleType()),
        StructField("user_segment", StringType()),
        StructField("payment_method", StringType()),
        StructField("country", StringType()),
        StructField("rolling_mean_amount_5", DoubleType()),
        StructField("rolling_std_amount_5", DoubleType()),
        StructField("rolling_max_amount_5", DoubleType()),
        StructField("rolling_mean_amount_10", DoubleType()),
        StructField("rolling_std_amount_10", DoubleType()),
        StructField("rolling_max_amount_10", DoubleType()),
        StructField("rolling_mean_amount_30", DoubleType()),
        StructField("rolling_std_amount_30", DoubleType()),
        StructField("rolling_max_amount_30", DoubleType()),
        StructField("amount_deviation_5", DoubleType()),
        StructField("amount_deviation_10", DoubleType()),
        StructField("amount_deviation_30", DoubleType())
    ])

    @pandas_udf(DoubleType())
    def predict_anomaly(pdf: pd.DataFrame) -> pd.Series:
        try:
            model = broadcast_vars['model'].value
            scaler = broadcast_vars['scaler'].value
            encoder = broadcast_vars['encoder'].value
            payment_columns = broadcast_vars['payment_columns'].value
            country_columns = broadcast_vars['country_columns'].value
            threshold = broadcast_vars['threshold'].value

            # Encodages
            payment_df = pd.get_dummies(pdf['payment_method'])
            for col in payment_columns:
                if col not in payment_df.columns:
                    payment_df[col] = 0
            payment_df = payment_df[payment_columns]

            country_df = pd.get_dummies(pdf['country'])
            for col in country_columns:
                if col not in country_df.columns:
                    country_df[col] = 0
            country_df = country_df[country_columns]

            pdf = pd.concat([pdf, payment_df, country_df], axis=1)
            segment_encoded = pd.DataFrame(
                encoder.transform(pdf[['user_segment']]),
                columns=['user_segment_encoded'],
                index=pdf.index
            )
            pdf = pd.concat([pdf, segment_encoded], axis=1)

            pdf['high_value'] = (pdf['amount'] > pdf['amount'].quantile(0.98)).astype(int)
            pdf['is_fast_transaction'] = (pdf['time_since_last'] < 60).astype(int)
            pdf['is_night'] = (pdf['hour_of_day'].between(0, 6)).astype(int)
            pdf['high_risk_combo'] = (
                (pdf['amount'] > pdf['amount'].quantile(0.95)) &
                (pdf['is_night'] == 1) &
                (pdf['is_fast_transaction'] == 1)
            ).astype(int)

            numeric_features = [
                'amount', 'quantity', 'hour_of_day', 'day_of_week', 'month',
                'time_since_last', 'ip_geodist', 'amount_per_item',
                'rolling_mean_amount_5', 'rolling_std_amount_5', 'rolling_max_amount_5',
                'rolling_mean_amount_10', 'rolling_std_amount_10', 'rolling_max_amount_10',
                'rolling_mean_amount_30', 'rolling_std_amount_30', 'rolling_max_amount_30',
                'amount_deviation_5', 'amount_deviation_10', 'amount_deviation_30'
            ]

            binary_features = [
                'user_segment_encoded', 'apple_pay', 'credit_card', 'debit_card', 
                'google_pay', 'paypal', 'Australia', 'Canada', 'France', 
                'Germany', 'UK', 'USA', 'high_value', 'is_fast_transaction', 
                'is_night', 'high_risk_combo'
            ]

            features = numeric_features + binary_features
            for feature in features:
                if feature not in pdf.columns:
                    pdf[feature] = 0  # Ajouter si manquant

            X = pdf[features].fillna(0)
            X_numeric = scaler.transform(X[numeric_features])
            X_processed = np.hstack([X_numeric, X[binary_features].values])
            anomaly_scores = model.decision_function(X_processed)
            predictions = (anomaly_scores < threshold).astype(int)

            return pd.Series(predictions)
        except Exception as e:
            print(f"Erreur pendant la prédiction : {e}")
            return pd.Series([0] * len(pdf), dtype='int')

    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka:29092") \
        .option("subscribe", "ecommerce") \
        .option("startingOffsets", "latest") \
        .option("failOnDataLoss", "false") \
        .load()

    transactions_df = kafka_df.selectExpr("CAST(value AS STRING) as json_string") \
        .select(from_json(col("json_string"), schema).alias("data")) \
        .select("data.*") \
        .withColumn("timestamp", current_timestamp())  # Utiliser timestamp actuel

    
    result_df = transactions_df.withColumn(
        "is_anomaly",
        predict_anomaly(struct([col(c.name) for c in schema.fields]))
    )
    # Fonction pour écrire chaque batch dans PostgreSQL
    def write_to_postgres(batch_df, batch_id):
        try:        
            # 1. Extraire les informations des utilisateurs et les dédupliquer
            users_df = batch_df.select("user_id", "user_segment", "country").distinct()
            users_data = users_df.collect()
            
            if users_data:  # Seulement si on a des utilisateurs
                conn = get_db_connection()
                cursor = conn.cursor()
                
                for row in users_data:
                    cursor.execute("""
                        INSERT INTO users (user_id, user_segment, country) 
                        VALUES (%s, %s, %s)
                        ON CONFLICT (user_id) DO NOTHING
                    """, (row['user_id'], row['user_segment'], row['country']))
                
                conn.commit()
                print(f"👥 Batch {batch_id}: {len(users_data)} utilisateurs traités")
                cursor.close()
                conn.close()
            # 2. Convertir la prédiction en boolean (true/false) au lieu de 1/0
            # Préparer les transactions avec la colonne is_anomaly convertie en boolean
            transactions_df = batch_df.select(
                "transaction_id", "user_id", "timestamp", "amount", "quantity", 
                "hour_of_day", "day_of_week", "month", "time_since_last", 
                "ip_geodist", "amount_per_item", "payment_method",
                (col("is_anomaly") > 0).cast("boolean").alias("is_anomaly")
            )
            
            # 3. Insérer d'abord les utilisateurs avec ON CONFLICT DO NOTHING
            # pour respecter la contrainte de clé étrangère
            
            # 4. Ensuite insérer les transactions
            transactions_df.write \
                .format("jdbc") \
                .option("url", "jdbc:postgresql://timescaledb:5432/anomalie_detection") \
                .option("dbtable", "transactions") \
                .option("user", "user") \
                .option("password", "password") \
                .option("driver", "org.postgresql.Driver") \
                .option("batchsize", 1000) \
                .mode("append") \
                .save()
            
            # Compter les anomalies dans ce batch
            anomaly_count = transactions_df.filter(col("is_anomaly") == True).count()
            total_count = transactions_df.count()
            
            print(f"✅ Batch {batch_id}: {total_count} transactions écrites (dont {anomaly_count} anomalies)")
            print(f"🕐 Timestamp utilisé: heure actuelle pour compatibilité Grafana")
            
        except Exception as e:
            print(f"Erreur lors de l'écriture dans PostgreSQL: {e}")

    # Console output pour le débogage
    console_query = result_df.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", "false") \
        .start()

    # PostgreSQL output 
    postgres_query = result_df.writeStream \
        .outputMode("append") \
        .foreachBatch(write_to_postgres) \
        .start()
    
    print("📊 Surveillance des transactions en cours...")
    print("Consommateur Spark lancé. Affichage des transactions...")
    console_query.awaitTermination()

  
if __name__ == "__main__":
    main()