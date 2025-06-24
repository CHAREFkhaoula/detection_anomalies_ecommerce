# Fonction pour écrire chaque batch dans PostgreSQL
    def write_to_postgres(batch_df, batch_id):
        try:
            # 1. Extraire les informations des utilisateurs et les dédupliquer
            users_df = batch_df.select("user_id", "user_segment", "country").distinct()
            
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
            users_df.write \
                .format("jdbc") \
                .option("url", "jdbc:postgresql://timescaledb:5432/anomalie_detection") \
                .option("dbtable", "users") \
                .option("user", "user") \
                .option("password", "password") \
                .option("driver", "org.postgresql.Driver") \
                .option("batchsize", 10) \
                .mode("append") \
                .save()
                
            # 4. Ensuite insérer les transactions
            transactions_df.write \
                .format("jdbc") \
                .option("url", "jdbc:postgresql://timescaledb:5432/anomalie_detection") \
                .option("dbtable", "transactions") \
                .option("user", "user") \
                .option("password", "password") \
                .option("driver", "org.postgresql.Driver") \
                .option("batchsize", 10) \
                .mode("append") \
                .save()
            
            print(f"Batch {batch_id}: {transactions_df.count()} transactions écrites dans PostgreSQL")
            
        except Exception as e:
            print(f"Erreur lors de l'écriture dans PostgreSQL: {e}")