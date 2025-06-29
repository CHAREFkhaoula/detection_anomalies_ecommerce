"""
SYSTÈME DE DÉPLOIEMENT AUTOMATIQUE MLOps E-COMMERCE
====================================================
Pipeline complet automatisé pour l'apprentissage :
- Preprocessing automatique des données
- Entraînement avec MLflow tracking  
- Déploiement API REST avec Flask
- Monitoring et gestion des versions
- Interface de validation automatique
"""

import os
import sys
import json
import time
import argparse
import threading
import subprocess
from datetime import datetime
from pathlib import Path

try:
    import mlflow
    import mlflow.sklearn
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score, fbeta_score
    import joblib
    from flask import Flask, request, jsonify, render_template_string
    import warnings
    warnings.filterwarnings('ignore')
    
    print("✅ Tous les modules importés avec succès")
    
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("📦 Installation des dépendances...")
    subprocess.run([sys.executable, "-m", "pip", "install", "mlflow", "scikit-learn", "pandas", "numpy", "flask", "joblib"], check=True)
    print("✅ Dépendances installées, relancez le script")
    sys.exit(1)

# ============================================================================
# 1. CLASSE PRINCIPALE DE DÉPLOIEMENT AUTOMATIQUE
# ============================================================================

class AutoMLOpsDeployment:
    def __init__(self):
        print("********* INITIALISATION SYSTÈME DÉPLOIEMENT AUTOMATIQUE MLOps")
        print("="*60)
        
        # Configuration des chemins
        self.base_dir = Path.cwd()
        self.models_dir = self.base_dir / "deployed_models"
        self.logs_dir = self.base_dir / "deployment_logs"
        self.mlruns_dir = self.base_dir / "mlruns"
        
        # Créer les répertoires
        for dir_path in [self.models_dir, self.logs_dir, self.mlruns_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Configuration MLflow
        mlflow.set_tracking_uri(f"file://{self.mlruns_dir}")
        mlflow.set_experiment("auto_mlops_ecommerce_deployment")
        
        # Composants ML
        self.model = None
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        self.threshold = -0.2
        self.model_metadata = {}
        self.deployment_status = {
            "last_training": None,
            "model_version": None,
            "api_status": "stopped",
            "performance": {}
        }
        
        # Configuration Flask
        self.app = Flask(__name__)
        self.setup_flask_routes()
        
        print(f"Système initialisé")
        print(f"Répertoire de travail: {self.base_dir}")
        print(f"MLflow UI: mlflow ui --backend-store-uri file://{self.mlruns_dir}")
        print(f"🌐 API sera disponible sur: http://localhost:5001")
        
    # ========================================================================
    # 2. GÉNÉRATION DE DONNÉES ET PREPROCESSING
    # ========================================================================
    
    def generate_sample_data(self, n_samples=1000000, anomaly_rate=0.02):
        """
        Générer des données d'exemple e-commerce avec anomalies
        """
        print(f"📊 Génération de {n_samples} échantillons (taux anomalie: {anomaly_rate:.1%})")
        
        # np.random.seed(42) 
        
        # Features de base
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='H')
        
        data = {
            'timestamp': timestamps,
            'amount': np.random.exponential(50, n_samples),
            'quantity': np.random.poisson(2, n_samples) + 1,
            'payment_method': np.random.choice([
                'credit_card', 'paypal', 'debit_card', 'apple_pay', 'google_pay'
            ], n_samples),
            'country': np.random.choice([
                'USA', 'Canada', 'UK', 'France', 'Germany', 'Australia'
            ], n_samples),
            'user_segment': np.random.choice([
                'low_spender', 'medium_spender', 'high_spender', 'vip'
            ], n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Features dérivées temporelles
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Features de comportement
        df['time_since_last'] = np.random.exponential(300, n_samples)
        df['ip_geodist'] = np.random.exponential(100, n_samples)
        df['amount_per_item'] = df['amount'] / df['quantity']
        
        # Rolling features (simulées)
        window_sizes = [5, 10, 30]
        for window in window_sizes:
            df[f'rolling_mean_amount_{window}'] = df['amount'].rolling(window, min_periods=1).mean()
            df[f'rolling_std_amount_{window}'] = df['amount'].rolling(window, min_periods=1).std().fillna(0)
            df[f'rolling_max_amount_{window}'] = df['amount'].rolling(window, min_periods=1).max()
            df[f'amount_deviation_{window}'] = np.abs(df['amount'] - df[f'rolling_mean_amount_{window}'])
        
        # Créer des anomalies synthétiques
        n_anomalies = int(anomaly_rate * n_samples)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        
        df['is_anomaly'] = 0
        df.loc[anomaly_indices, 'is_anomaly'] = 1
        
        # Rendre les anomalies détectables
        df.loc[anomaly_indices, 'amount'] *= np.random.uniform(5, 15, n_anomalies) # augmenter 
        df.loc[anomaly_indices, 'time_since_last'] *= np.random.uniform(0.05, 0.2, n_anomalies) # reduire
        df.loc[anomaly_indices, 'quantity'] *= np.random.uniform(3, 8, len(anomaly_indices)) # augmenter
        
        print(f"Données générées ✅ : {len(df)} transactions, {n_anomalies} anomalies")
        return df
    
    def preprocess_data(self, df, is_training=True):
        """
        Preprocessing complet des données
        """
        print(f"Preprocessing des données ({'training' if is_training else 'prediction'})...")
        
        # Assurer que df est un DataFrame pandas
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        # Gérer les features temporelles: si 'timestamp' est présent, dériver; sinon, supposer qu'elles sont déjà là
        if 'timestamp' in df.columns:
            df['hour_of_day'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
        else:
            # S'assurer que les colonnes temporelles existent, sinon les initialiser à 0 ou NaN
            for col in ['hour_of_day', 'day_of_week', 'month']:
                if col not in df.columns:
                    df[col] = 0 # Ou np.nan, selon la gestion des valeurs manquantes souhaitée

        # Gérer les features de comportement (si non présentes, initialiser)
        for col in ['time_since_last', 'ip_geodist']:
            if col not in df.columns:
                df[col] = 0

        # Calculer amount_per_item, gérer le cas de division par zéro
        df['amount_per_item'] = df['amount'] / df['quantity']
        df['amount_per_item'] = df['amount_per_item'].replace([np.inf, -np.inf], 0).fillna(0)

        # Gérer les rolling features (simulées pour la prédiction si non présentes)
        window_sizes = [5, 10, 30]

        for window in window_sizes:
            for feature_suffix in ['mean', 'std', 'max']:
                col_name = f'rolling_{feature_suffix}_amount_{window}'
                if col_name not in df.columns:
                    if 'amount' in df.columns:
                        if feature_suffix == 'mean':
                            df[col_name] = df['amount'].rolling(window).mean().fillna(0)
                        elif feature_suffix == 'std':
                            df[col_name] = df['amount'].rolling(window).std().fillna(0)
                        elif feature_suffix == 'max':
                            df[col_name] = df['amount'].rolling(window).max().fillna(0)
                    else:
                        df[col_name] = 0  # fallback si amount absent
                        print("************",df[col_name])

            col_name_dev = f'amount_deviation_{window}'
            if col_name_dev not in df.columns:
                if f'rolling_mean_amount_{window}' in df.columns and 'amount' in df.columns:
                    df[col_name_dev] = (df['amount'] - df[f'rolling_mean_amount_{window}']).fillna(0)
                else:
                    df[col_name_dev] = 0

        # Features catégorielles - Payment methods
        payment_methods = ['apple_pay', 'credit_card', 'debit_card', 'google_pay', 'paypal']
        # Assurer que 'payment_method' est une colonne string pour get_dummies
        if 'payment_method' in df.columns:
            df['payment_method'] = df['payment_method'].astype(str)
            payment_df = pd.get_dummies(df['payment_method'], prefix='')
        else:
            payment_df = pd.DataFrame(0, index=df.index, columns=payment_methods)

        for method in payment_methods:
            if method not in payment_df.columns:
                payment_df[method] = 0
        payment_df = payment_df[payment_methods]
        
        # Features catégorielles - Countries
        countries = ['Australia', 'Canada', 'France', 'Germany', 'UK', 'USA']
        # Assurer que 'country' est une colonne string pour get_dummies
        if 'country' in df.columns:
            df['country'] = df['country'].astype(str)
            country_df = pd.get_dummies(df['country'], prefix='')
        else:
            country_df = pd.DataFrame(0, index=df.index, columns=countries)

        for country in countries:
            if country not in country_df.columns:
                country_df[country] = 0
        country_df = country_df[countries]
        
        # Encoder les segments utilisateur
        if is_training:
            # Entraîner l'encoder
            # Assurer que 'user_segment' est une colonne string
            df['user_segment'] = df['user_segment'].astype(str)
            self.encoder.fit(df[['user_segment']])
            # Sauvegarder les colonnes pour plus tard
            self.payment_columns = payment_methods
            self.country_columns = countries
        
        # Appliquer l'encodage
        # Gérer le cas où 'user_segment' est absent ou contient des valeurs inconnues
        if 'user_segment' in df.columns:
            # Convertir en string pour l'encoder
            df['user_segment'] = df['user_segment'].astype(str)
            # Utiliser transform, et gérer les valeurs inconnues si elles apparaissent
            # L'encoder de sklearn lève une erreur pour les valeurs inconnues par défaut
            # Une approche plus robuste serait d'utiliser OneHotEncoder avec handle_unknown='ignore'
            # Pour cet exemple, nous allons simplement s'assurer que la colonne existe
            try:
                segment_encoded = self.encoder.transform(df[['user_segment']])
            except ValueError:
                # Si des segments inconnus apparaissent, les traiter comme une catégorie par défaut (ex: 0)
                # Cela dépend de la stratégie de gestion des catégories inconnues
                print("Segments utilisateur inconnus détectés. Traitement comme catégorie par défaut.")
                # Créer un mapping pour les segments connus
                known_segments = list(self.encoder.classes_)
                segment_mapping = {s: self.encoder.transform([[s]])[0][0] for s in known_segments}
                # Appliquer le mapping, et assigner une valeur par défaut aux inconnus
                segment_encoded = df['user_segment'].apply(lambda x: segment_mapping.get(x, 0)).values.reshape(-1, 1)

        else:
            segment_encoded = np.zeros((len(df), 1)) # Si la colonne est absente, initialiser à 0

        segment_df = pd.DataFrame(
            segment_encoded,
            columns=['user_segment_encoded'],
            index=df.index
        )
        
        # Combiner toutes les features
        df_processed = pd.concat([df, payment_df, country_df, segment_df], axis=1)
        
        # Features binaires dérivées
        # Assurer que 'amount' et 'quantity' sont numériques
        df_processed['amount'] = pd.to_numeric(df_processed['amount'], errors='coerce').fillna(0)
        df_processed['quantity'] = pd.to_numeric(df_processed['quantity'], errors='coerce').fillna(1) # quantity ne peut pas être 0

        # Gérer le cas où le quantile n'est pas calculable (ex: df vide ou trop peu de données)
        if not df_processed['amount'].empty:
            high_value_threshold = df_processed['amount'].quantile(0.98)
        else:
            high_value_threshold = 0 # Valeur par défaut si pas de données

        df_processed['high_value'] = (df_processed['amount'] > high_value_threshold).astype(int)
        df_processed['is_fast_transaction'] = (df_processed['time_since_last'] < 60).astype(int)
        df_processed['is_night'] = (df_processed['hour_of_day'].between(0, 6)).astype(int)
        
        # Gérer le cas où le quantile n'est pas calculable pour high_risk_combo
        if not df_processed['amount'].empty:
            high_risk_amount_threshold = df_processed['amount'].quantile(0.95)
        else:
            high_risk_amount_threshold = 0

        df_processed['high_risk_combo'] = (
            (df_processed['amount'] > high_risk_amount_threshold) &
            (df_processed['is_night'] == 1) &
            (df_processed['is_fast_transaction'] == 1)
        ).astype(int)
        
        # Sélection des features finales
        numeric_features = [
            'amount', 'quantity', 'hour_of_day', 'day_of_week', 'month',
            'time_since_last', 'ip_geodist', 'amount_per_item',
            'rolling_mean_amount_5', 'rolling_std_amount_5', 'rolling_max_amount_5',
            'rolling_mean_amount_10', 'rolling_std_amount_10', 'rolling_max_amount_10',
            'rolling_mean_amount_30', 'rolling_std_amount_30', 'rolling_max_amount_30',
            'amount_deviation_5', 'amount_deviation_10', 'amount_deviation_30'
        ]
        
        binary_features = [
            'user_segment_encoded'
        ] + payment_methods + countries + [
            'high_value', 'is_fast_transaction', 'is_night', 'high_risk_combo'
        ]
        
        # Vérifier que toutes les features existent et les initialiser si elles manquent
        all_features = numeric_features + binary_features
        for feature in all_features:
            if feature not in df_processed.columns:
                df_processed[feature] = 0
                print(f"Feature manquante ajoutée: {feature}")
        
        # Preprocessing final
        X = df_processed[all_features].fillna(0)
        
        if is_training:
            # Entraîner le scaler
            X_numeric_scaled = self.scaler.fit_transform(X[numeric_features])
        else:
            # Appliquer le scaler pré-entraîné
            # Assurer que le scaler a été entraîné
            if not hasattr(self.scaler, 'scale_'):
                raise RuntimeError("Scaler non entraîné. Entraînez un modèle d'abord.")
            X_numeric_scaled = self.scaler.transform(X[numeric_features])
        
        # Combiner features numériques et binaires
        X_final = np.hstack([X_numeric_scaled, X[binary_features].values])
        
        print(f"✅ Preprocessing terminé: {X_final.shape[0]} échantillons, {X_final.shape[1]} features")
        
        return X_final, all_features
    
    # ========================================================================
    # 3. ENTRAÎNEMENT AUTOMATIQUE
    # ========================================================================
    
    def auto_train_model(self):
        """
        Pipeline d'entraînement automatique
        """
        print("DÉBUT ENTRAÎNEMENT AUTOMATIQUE")
        print("="*50)
        
        try:
            with mlflow.start_run(run_name=f"auto_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
                
                # 1. Générer/Charger les données
                print("Étape 1: Chargement des données...")
                df = self.generate_sample_data()
                
                # 2. Preprocessing
                print("Étape 2: Preprocessing...")
                X, feature_names = self.preprocess_data(df, is_training=True)
                y = df['is_anomaly'].values
                
                # 3. Configuration du modèle
                contamination_rate = y.mean()
                model_params = {
                    'contamination': contamination_rate,
                    'n_estimators': 1000,
                    'random_state': 42,
                    'n_jobs': -1,
                    'max_samples': 'auto'
                }
                
                # Logger les paramètres
                mlflow.log_params(model_params)
                mlflow.log_param("n_samples", X.shape[0])
                mlflow.log_param("n_features", X.shape[1])
                mlflow.log_param("anomaly_rate", contamination_rate)
                
                # 4. Entraînement
                print("Étape 3: Entraînement du modèle...")
                self.model = IsolationForest(**model_params)
                self.model.fit(X)
                
                # 5. Optimisation du threshold
                print("Étape 4: Optimisation du threshold...")
                self.threshold = self._optimize_threshold(X, y)
                
                # 6. Évaluation finale
                print("Étape 5: Évaluation finale...")
                metrics = self._evaluate_model(X, y)
                
                # Logger les métriques
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                mlflow.log_metric("optimal_threshold", self.threshold)
                
                # 7. Sauvegarde du modèle
                print("Étape 6: Sauvegarder notre modele...")
                model_version = self._save_model_for_deployment(run.info.run_id, metrics)
                
                # Logger le modèle dans MLflow
                mlflow.sklearn.log_model(self.model, "isolation_forest_model")
                
                # 8. Mise à jour du statut
                self.deployment_status.update({
                    "last_training": datetime.now().isoformat(),
                    "model_version": model_version,
                    "performance": metrics,
                    "run_id": run.info.run_id
                })
                
                print(f"ENTRAÎNEMENT RÉUSSI !")
                print(f"📋 Résumé:")
                print(f"   • Run ID: {run.info.run_id}")
                print(f"   • AUC Score: {metrics.get('auc_score', 0):.3f}")
                print(f"   • Precision: {metrics.get('precision', 0):.3f}")
                print(f"   • Recall: {metrics.get('recall', 0):.3f}")
                print(f"   • Threshold: {self.threshold:.4f}")
                print(f"   • Version: {model_version}")
                
                return True, run.info.run_id, metrics
                
        except Exception as e:
            print(f"Erreur pendant l'entraînement: {e}")
            import traceback
            traceback.print_exc()
            return False, None, {}
    
    def _optimize_threshold(self, X, y_true):
        """Optimiser le threshold pour maximiser F-beta score"""
        y_scores = self.model.decision_function(X)
        thresholds = np.linspace(y_scores.min(), y_scores.max(), 200)
        
        best_score = 0
        best_threshold = -0.2
        
        for threshold in thresholds:
            y_pred = (y_scores < threshold).astype(int)
            
            try:
                f_beta = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
                if f_beta > best_score:
                    best_score = f_beta
                    best_threshold = threshold
            except:
                continue
        
        return best_threshold
    
    def _evaluate_model(self, X, y_true):
        """Évaluer les performances du modèle"""
        y_scores = self.model.decision_function(X)
        y_pred = (y_scores < self.threshold).astype(int)
        
        try:
            metrics = {
                'auc_score': roc_auc_score(y_true, -y_scores),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'accuracy': (y_pred == y_true).mean(),
                'anomaly_rate_predicted': y_pred.mean(),
                'anomaly_rate_actual': y_true.mean()
            }
        except Exception as e:
            print(f"Erreur calcul métriques: {e}")
            metrics = {
                'auc_score': 0.5,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'accuracy': 0.0,
                'anomaly_rate_predicted': 0.0,
                'anomaly_rate_actual': 0.0
            }
        
        return metrics
    
    def _save_model_for_deployment(self, run_id, metrics):
        """Sauvegarder le modèle prêt pour le déploiement"""
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.models_dir / f"model_v{version}"
        model_dir.mkdir(exist_ok=True)
        
        # Sauvegarder tous les composants
        joblib.dump(self.model, model_dir / "model.joblib")
        joblib.dump(self.scaler, model_dir / "scaler.joblib")
        joblib.dump(self.encoder, model_dir / "encoder.joblib")
        
        # Métadonnées complètes
        metadata = {
            "version": version,
            "run_id": run_id,
            "trained_at": datetime.now().isoformat(),
            "threshold": float(self.threshold),
            "payment_columns": self.payment_columns,
            "country_columns": self.country_columns,
            "performance": metrics,
            "model_params": {
                "contamination": self.model.contamination,
                "n_estimators": self.model.n_estimators,
                "n_features": self.model.n_features_in_
            }
        }
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Marquer comme modèle de production
        production_link = self.models_dir / "production"
        if production_link.exists():
            production_link.unlink()
        production_link.symlink_to(model_dir.name)
        
        self.model_metadata = metadata
        
        print(f"Modèle sauvé: {model_dir}")
        return version
    
    # ========================================================================
    # 4. API REST FLASK POUR LE DÉPLOIEMENT
    # ========================================================================
    
    def setup_flask_routes(self):
        """Configurer les routes Flask pour l'API"""
        
        @self.app.route('/')
        def dashboard():
            """Dashboard principal de monitoring"""
            return render_template_string(self._get_dashboard_template(), 
                                        status=self.deployment_status)
        
        @self.app.route('/health')
        def health_check():
            """Check de santé de l'API"""
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "model_loaded": self.model is not None,
                "deployment_info": self.deployment_status
            })
        
        @self.app.route('/predict', methods=['POST'])
        def predict_anomaly():
            """Endpoint de prédiction principal"""
            try:
                if self.model is None:
                    return jsonify({
                        "error": "Modèle non chargé. Entraînez d'abord un modèle.",
                        "success": False
                    }), 400
                
                # Récupérer les données
                data = request.get_json()
                
                if 'transactions' not in data:
                    return jsonify({
                        "error": "Format attendu: {'transactions': [transaction_data]}",
                        "success": False
                    }), 400
                
                # Convertir en DataFrame
                df = pd.DataFrame(data['transactions'])
                
                # Preprocessing
                X, _ = self.preprocess_data(df, is_training=False)
                # is_training=False = on indique qu’on est en phase de prédiction, pas d'entraînement

                # Prédictions
                scores = self.model.decision_function(X)
                anomalies = (scores < self.threshold).astype(int)
                probabilities = 1 / (1 + np.exp(scores))  # Sigmoid approximation
                
                # Résultats détaillés
                results = []
                for i, (score, is_anomaly, prob) in enumerate(zip(scores, anomalies, probabilities)):
                    results.append({
                        "transaction_id": i,
                        "is_anomaly": bool(is_anomaly),
                        "anomaly_score": float(score),
                        "anomaly_probability": float(prob),
                        "risk_level": self._get_risk_level(prob)
                    })
                
                return jsonify({
                    "success": True,
                    "predictions": results,
                    "summary": {
                        "total_transactions": len(results),
                        "anomalies_detected": int(anomalies.sum()),
                        "anomaly_rate": float(anomalies.mean()),
                        "model_version": self.deployment_status.get("model_version", "unknown")
                    },
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                return jsonify({
                    "error": f"Erreur de prédiction: {str(e)}",
                    "success": False
                }), 500
        
        @self.app.route('/retrain', methods=['POST'])
        def trigger_retrain():
            """Déclencher un réentraînement"""
            try:
                print("🔄 Réentraînement déclenché via API...")
                success, run_id, metrics = self.auto_train_model()
                
                if success:
                    return jsonify({
                        "success": True,
                        "message": "Réentraînement réussi",
                        "run_id": run_id,
                        "metrics": metrics
                    })
                else:
                    return jsonify({
                        "success": False,
                        "message": "Échec du réentraînement"
                    }), 500
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/models')
        def list_models():
            """Lister les modèles disponibles"""
            models = []
            for model_dir in self.models_dir.iterdir():
                if model_dir.is_dir() and model_dir.name != 'production':
                    metadata_file = model_dir / "metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                        models.append(metadata)
            
            return jsonify({
                "models": sorted(models, key=lambda x: x['trained_at'], reverse=True),
                "production_model": self.deployment_status.get("model_version")
            })
    
    def _get_risk_level(self, probability):
        """Déterminer le niveau de risque"""
        if probability > 0.8:
            return "HIGH"
        elif probability > 0.6:
            return "MEDIUM"
        elif probability > 0.4:
            return "LOW"
        else:
            return "NORMAL"
    
    def _get_dashboard_template(self):
        """Template HTML plus joli et professionnel pour le dashboard"""
        return '''
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>MLOps Auto-Deploy Dashboard</title>
            <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
            <style>
                :root {
                    --primary-color: #6A057F; /* Violet profond */
                    --secondary-color: #883988; /* Violet plus clair */
                    --accent-color-1: #FF6B6B; /* Rouge corail */
                    --accent-color-2: #FFD166; /* Jaune doux */
                    --text-color: #333;
                    --light-text-color: #555;
                    --bg-color: #f0f2f5;
                    --card-bg: #ffffff;
                    --border-color: #e0e0e0;
                    --shadow-light: rgba(0, 0, 0, 0.08);
                    --shadow-medium: rgba(0, 0, 0, 0.15);
                    --success-color: #28a745;
                    --warning-color: #ffc107;
                    --error-color: #dc3545;
                }

                body {
                    font-family: 'Poppins', sans-serif;
                    margin: 0;
                    padding: 40px 20px;
                    background: linear-gradient(135deg, var(--bg-color) 0%, #e9ecef 100%);
                    color: var(--text-color);
                    line-height: 1.6;
                    min-height: 100vh;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }

                .container {
                    max-width: 1200px;
                    width: 100%;
                    margin: 0 auto;
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 30px;
                }

                h1 {
                    font-size: 3em;
                    color: var(--primary-color);
                    text-align: center;
                    margin-bottom: 50px;
                    font-weight: 700;
                    letter-spacing: -1px;
                    text-shadow: 2px 2px 4px var(--shadow-light);
                    background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }

                .card {
                    background: var(--card-bg);
                    padding: 30px;
                    border-radius: 15px;
                    box-shadow: 0 10px 25px var(--shadow-medium);
                    border: 1px solid var(--border-color);
                    display: flex;
                    flex-direction: column;
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                }

                .card:hover {
                    transform: translateY(-5px);
                    box-shadow: 0 15px 35px var(--shadow-medium);
                }

                .card h2 {
                    font-size: 1.8em;
                    color: var(--primary-color);
                    margin-top: 0;
                    margin-bottom: 25px;
                    padding-bottom: 12px;
                    border-bottom: 3px solid var(--accent-color-1);
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    font-weight: 600;
                    text-transform: uppercase;
                }

                .card p {
                    margin-bottom: 12px;
                    font-size: 1.1em;
                    color: var(--light-text-color);
                }

                .card p strong {
                    color: var(--text-color);
                    font-weight: 600;
                }

                .status-good { color: var(--success-color); font-weight: 700; }
                .status-warning { color: var(--warning-color); font-weight: 700; }
                .status-error { color: var(--error-color); font-weight: 700; }

                .metrics-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }

                .metric-item {
                    background: linear-gradient(45deg, #f8f9fa, #e9ecef);
                    padding: 20px;
                    border-radius: 10px;
                    border: 1px solid var(--border-color);
                    text-align: center;
                    box-shadow: 0 4px 10px var(--shadow-light);
                    transition: background 0.3s ease;
                }

                .metric-item:hover {
                    background: linear-gradient(45deg, #e9ecef, #f8f9fa);
                }

                .metric-item strong {
                    display: block;
                    font-size: 0.95em;
                    color: var(--light-text-color);
                    margin-bottom: 8px;
                }

                .metric-value {
                    font-size: 1.8em;
                    font-weight: 700;
                    color: var(--accent-color-1);
                }

                .button-group {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin-top: 30px;
                    justify-content: center;
                }

                .btn {
                    background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
                    color: white;
                    padding: 15px 25px;
                    border: none;
                    border-radius: 10px;
                    cursor: pointer;
                    font-size: 1.1em;
                    font-weight: 600;
                    transition: all 0.3s ease;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    box-shadow: 0 5px 15px var(--shadow-medium);
                }

                .btn:hover {
                    background: linear-gradient(45deg, var(--secondary-color), var(--primary-color));
                    transform: translateY(-3px) scale(1.02);
                    box-shadow: 0 8px 20px var(--shadow-medium);
                }

                .btn:active {
                    transform: translateY(0);
                    box-shadow: 0 3px 10px var(--shadow-light);
                }

                pre {
                    background: #2d2d2d;
                    color: #f8f8f2;
                    padding: 20px;
                    border-radius: 10px;
                    overflow-x: auto;
                    font-family: 'Fira Code', 'Consolas', 'Monaco', monospace;
                    font-size: 0.95em;
                    margin-top: 25px;
                    box-shadow: 0 5px 15px var(--shadow-medium);
                }

                code {
                    font-family: 'Fira Code', 'Consolas', 'Monaco', monospace;
                }

                /* Responsive adjustments */
                @media (max-width: 768px) {
                    body {
                        padding: 20px 15px;
                    }
                    .container {
                        grid-template-columns: 1fr;
                        gap: 20px;
                    }
                    h1 {
                        font-size: 2.2em;
                        margin-bottom: 30px;
                    }
                    .card {
                        padding: 20px;
                    }
                    .card h2 {
                        font-size: 1.5em;
                        margin-bottom: 15px;
                    }
                    .btn {
                        width: 100%;
                        justify-content: center;
                        padding: 12px 20px;
                    }
                    .metrics-grid {
                        grid-template-columns: 1fr;
                    }
                }
            </style>
        </head>
        <body>
            <h1><i class="fas fa-brain"></i> MLOps Dashboard</h1>
            
            <div class="container">
                <div class="card">
                    <h2><i class="fas fa-info-circle"></i> État du Système</h2>
                    <p><strong>Statut API:</strong> 
                        <span class="status-good">{{ status.api_status|title }}</span>
                    </p>
                    <p><strong>Dernier Entraînement:</strong> 
                        {{ status.last_training or 'Aucun' }}
                    </p>
                    <p><strong>Version du Modèle:</strong> 
                        {{ status.model_version or 'Aucune' }}
                    </p>
                </div>
                
                <div class="card">
                    <h2><i class="fas fa-chart-line"></i> Performances du Modèle</h2>
                    {% if status.performance %}
                    <div class="metrics-grid">
                        <div class="metric-item">
                            <strong>AUC Score:</strong>
                            <span class="metric-value">{{ "%.3f"|format(status.performance.auc_score or 0) }}</span>
                        </div>
                        <div class="metric-item">
                            <strong>Precision:</strong>
                            <span class="metric-value">{{ "%.3f"|format(status.performance.precision or 0) }}</span>
                        </div>
                        <div class="metric-item">
                            <strong>Recall:</strong>
                            <span class="metric-value">{{ "%.3f"|format(status.performance.recall or 0) }}</span>
                        </div>
                        <div class="metric-item">
                            <strong>F1 Score:</strong>
                            <span class="metric-value">{{ "%.3f"|format(status.performance.f1_score or 0) }}</span>
                        </div>
                    </div>
                    {% else %}
                    <p>Aucune métrique disponible - Entraînez un modèle d'abord</p>
                    {% endif %}
                </div>

                <div class="card">
                    <h2><i class="fas fa-cogs"></i> Actions Rapides</h2>
                    <div class="button-group">
                        <button class="btn" onclick="retrain()"><i class="fas fa-sync-alt"></i> Réentraîner le Modèle</button>
                        <button class="btn" onclick="testAPI()"><i class="fas fa-flask"></i> Test de l'API</button>
                        <button class="btn" onclick="viewModels()"><i class="fas fa-list-alt"></i> Voir les Modèles</button>
                    </div>
                </div>
                
                <div class="card" style="grid-column: 1 / -1;">
                    <h2><i class="fas fa-code"></i> Exemple d'Usage API</h2>
                    <pre><code>
# Test de prédiction
curl -X POST http://localhost:5001/predict \n-H "Content-Type: application/json" \n-d 
'{
  "transactions": [{
    "amount": 1500.0,
    "quantity": 1,
    "payment_method": "credit_card",
    "country": "USA",
    "user_segment": "vip",
    "hour_of_day": 2,
    "day_of_week": 6,
    "month": 12,
    "time_since_last": 30,
    "ip_geodist": 50,
    "amount_per_item": 1500.0,
    "rolling_mean_amount_5": 1000.0,
    "rolling_std_amount_5": 50.0,
    "rolling_max_amount_5": 1500.0,
    "rolling_mean_amount_10": 900.0,
    "rolling_std_amount_10": 60.0,
    "rolling_max_amount_10": 1500.0,
    "rolling_mean_amount_30": 800.0,
    "rolling_std_amount_30": 70.0,
    "rolling_max_amount_30": 1500.0,
    "amount_deviation_5": 500.0,
    "amount_deviation_10": 600.0,
    "amount_deviation_30": 700.0
  }]
}'
                    </code></pre>
                </div>
            </div>
            
            
            <script>
                function retrain( ) {
                    if(confirm('Déclencher un réentraînement ?')) {
                        fetch('/retrain', {method: 'POST'})
                        .then(response => response.json())
                        .then(data => {
                            alert(data.success ? 'Réentraînement réussi!' : 'Erreur: ' + data.message);
                            if(data.success) location.reload();
                        })
                        .catch(err => alert('Erreur lors du réentraînement: ' + err));
                    }
                }
                
                function testAPI() {
                    const testData = {
                        transactions: [{
                            amount: 1500.0, quantity: 1, payment_method: "credit_card",
                            country: "USA", user_segment: "vip", hour_of_day: 2,
                            day_of_week: 6, month: 12, time_since_last: 30, ip_geodist: 50,"amount_per_item": 1500.0,
    rolling_mean_amount_5: 1000.0,
    rolling_std_amount_5: 50.0,
    rolling_max_amount_5: 1500.0,
    rolling_mean_amount_10: 900.0,
    rolling_std_amount_10: 60.0,
    rolling_max_amount_10: 1500.0,
    rolling_mean_amount_30: 800.0,
    rolling_std_amount_30: 70.0,
    rolling_max_amount_30: 1500.0,
    amount_deviation_5: 500.0,
    amount_deviation_10: 600.0,
    amount_deviation_30: 700.0
                        }]
                    };
                    
                    fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(testData)
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success && data.predictions && data.predictions.length > 0) {
                            alert('Test API réussi! Anomalie détectée: ' + 
                                  (data.predictions[0].is_anomaly ? 'OUI' : 'NON') +
                                  ' (Score: ' + data.predictions[0].anomaly_score.toFixed(4) + ')');
                        } else {
                            alert('Test API réussi, mais réponse inattendue.');
                        }
                    })
                    .catch(err => alert('Erreur test API: ' + err));
                }
                
                function viewModels() {
                    fetch('/models')
                    .then(response => response.json())
                    .then(data => {
                        if (data.models && data.models.length > 0) {
                            const models = data.models.map(m => 
                                `Version: ${m.version} - AUC: ${m.performance.auc_score.toFixed(3)} - Entraîné le: ${new Date(m.trained_at).toLocaleString()}`
                            ).join('\\n');
                            alert('Modèles disponibles:\\n' + models + '\\n\\nModèle en production: ' + (data.production_model || 'Aucun'));
                        } else {
                            alert('Aucun modèle disponible.');
                        }
                    })
                    .catch(err => alert('Erreur lors de la récupération des modèles: ' + err));
                }
                
                // Auto-refresh toutes les 30 secondes
                setInterval(() => location.reload(), 30000);
            </script>
        </body>
        </html>
        '''
    
    # ========================================================================
    # 5. CHARGEMENT ET GESTION DES MODÈLES
    # ========================================================================
    
    def load_production_model(self):
        """Charger le modèle de production pour le déploiement"""
        production_path = self.models_dir / "production"
        
        if not production_path.exists():
            print("⚠️ Aucun modèle de production trouvé")
            return False
        
        try:
            # Charger les composants
            self.model = joblib.load(production_path / "model.joblib")
            self.scaler = joblib.load(production_path / "scaler.joblib")  
            self.encoder = joblib.load(production_path / "encoder.joblib")
            
            # Charger les métadonnées
            with open(production_path / "metadata.json") as f:
                self.model_metadata = json.load(f)
            
            self.threshold = self.model_metadata["threshold"]
            self.payment_columns = self.model_metadata["payment_columns"]
            self.country_columns = self.model_metadata["country_columns"]
            
            # Mettre à jour le statut
            self.deployment_status.update({
                "model_version": self.model_metadata["version"],
                "performance": self.model_metadata["performance"],
                "api_status": "ready"
            })
            
            print(f"Modèle de production chargé: {self.model_metadata['version']}")
            return True
            
        except Exception as e:
            print(f" Erreur chargement modèle: {e}")
            return False

   	# ========================================================================
    # 6. PIPELINE DE DÉPLOIEMENT AUTOMATIQUE
    # ========================================================================
    
    def run_auto_deployment_pipeline(self, train_first=True):
        """
        Pipeline complet de déploiement automatique
        """
        print("DÉMARRAGE PIPELINE DE DÉPLOIEMENT AUTOMATIQUE")
        print("="*60)
        
        try:
            # Étape 1: Entraînement (si demandé)
            if train_first:
                print("Phase 1: Entraînement automatique...")
                success, run_id, metrics = self.auto_train_model()
                
                if not success:
                    print("Échec de l'entraînement, arrêt du pipeline")
                    return False
                
                print(f"Entraînement réussi (Run: {run_id})")
            
            # Étape 2: Chargement du modèle
            print("Phase 2: Chargement du modèle de production...")
            if not self.load_production_model():
                print("Échec du chargement, arrêt du pipeline")
                return False
            
            # Étape 3: Validation du déploiement
            print("Phase 3: Validation du déploiement...")
            if not self._validate_deployment():
                print("Validation échouée, arrêt du pipeline")
                return False
            
            # Étape 4: Démarrage de l'API
            print("Phase 4: Démarrage de l'API de production...")
            print(f"DÉPLOIEMENT AUTOMATIQUE RÉUSSI !")
            print(f"Dashboard: http://localhost:5001")
            print(f"MLflow UI: mlflow ui --backend-store-uri file://{self.mlruns_dir}")
            print(f"\n💡 Utilisez Ctrl+C pour arrêter le serveur")
            
            # Démarrer le serveur Flask
            self.deployment_status["api_status"] = "running"
            self.app.run(host='0.0.0.0', port=5001, debug=False)
            
            return True
            
        except KeyboardInterrupt:
            print(f"\n🛑 Arrêt du serveur demandé par l'utilisateur")
            self.deployment_status["api_status"] = "stopped"
            return True
        except Exception as e:
            print(f"Erreur pipeline de déploiement: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _validate_deployment(self):
        """Valider que le déploiement fonctionne correctement"""
        try:
            # Test 1: Modèle chargé
            if self.model is None:
                print("Modèle non chargé")
                return False
            
            # Test 2: Métadonnées disponibles
            if not self.model_metadata:
                print("Métadonnées manquantes")
                return False
            
            # Test 3: Test de prédiction
            print("Test de prédiction...")
            test_data = self.generate_sample_data(n_samples=100)
            X_test, _ = self.preprocess_data(test_data, is_training=False)
            
            predictions = self.model.decision_function(X_test)
            anomalies = (predictions < self.threshold).sum()
            
            print(f"Test réussi: {anomalies}/100 anomalies détectées")
            
            # Test 4: Performances min 
            perf = self.model_metadata.get("performance", {})
            min_auc = 0.6  # Critère minimal pour la production
            
            if perf.get("auc_score", 0) < min_auc:
                print(f"Performance insuffisante: AUC {perf.get('auc_score', 0):.3f} < {min_auc}")
                return False
            
            print(f"✅ Performance validée: AUC {perf.get('auc_score', 0):.3f}")
            return True
            
        except Exception as e:
            print(f"Erreur validation: {e}")
            return False

# ============================================================================
# 7. INTERFACE EN LIGNE DE COMMANDE
# ============================================================================

def main():
    """Interface principale en ligne de commande"""
    parser = argparse.ArgumentParser(
        description="Système de déploiement automatique MLOps E-commerce",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Exemples d'utilisation:

  # Pipeline complet (entraînement + déploiement)
  python auto_mlops_deploy.py --full-deploy
  
  # Seulement entraîner un nouveau modèle  
  python auto_mlops_deploy.py --train-only
  
  # Déployer le modèle existant
  python auto_mlops_deploy.py --serve-only
  
  # Monitoring avec MLflow UI
  python auto_mlops_deploy.py --monitor
        '''
    )
    
    parser.add_argument('--full-deploy', action='store_true',
                       help='Pipeline complet: entraîner + déployer')
    parser.add_argument('--train-only', action='store_true',
                       help='Seulement entraîner un nouveau modèle')
    parser.add_argument('--serve-only', action='store_true',
                       help='Déployer le modèle existant')
    parser.add_argument('--monitor', action='store_true',
                       help='Lancer MLflow UI pour monitoring')
    
    args = parser.parse_args()
    
    # Si aucun argument, mode interactif
    if not any([args.full_deploy, args.train_only, args.serve_only, args.monitor]):
        print("**********SYSTÈME DE DÉPLOIEMENT AUTOMATIQUE MLOps*************")
        print("="*50)
        print("Choisissez une option:")
        print("1.  Pipeline complet (entraînement + déploiement)")
        print("2.  Entraînement seulement")
        print("3.  Déploiement seulement")
        print("4.  Monitoring MLflow")
        print("5.  Quitter")
        
        choice = input("\nVotre choix (1-5): ").strip()
        
        if choice == '1':
            args.full_deploy = True
        elif choice == '2':
            args.train_only = True
        elif choice == '3':
            args.serve_only = True
        elif choice == '4':
            args.monitor = True
        else:
            print("Au revoir! 👋")
            return
    
    # Initialiser le système
    deploy_system = AutoMLOpsDeployment()
    
    try:
        # Exécuter l'action demandée
        if args.monitor:
            print("📊 Lancement de MLflow UI...")
            mlruns_path = deploy_system.mlruns_dir
            subprocess.run([
                "mlflow", "ui", 
                "--backend-store-uri", f"file://{mlruns_path}",
                "--host", "0.0.0.0",
                "--port", "5000"
            ])
        
        elif args.train_only:
            print(" Mode: Entraînement seulement")
            success, run_id, metrics = deploy_system.auto_train_model()
            if success:
                print(f"\n Entraînement terminé avec succès!")
                print(f" Pour voir les résultats sur l'interface tapez --> mlflow ui")
            else:
                print(f"\n Échec de l'entraînement")
        
        elif args.serve_only:
            print(" Mode: Déploiement seulement")
            if deploy_system.load_production_model():
                deploy_system.run_auto_deployment_pipeline(train_first=False)
            else:
                print(" Aucun modèle à déployer. Entraînez d'abord un modèle.")
        
        elif args.full_deploy:
            print(" Mode: Pipeline complet")
            deploy_system.run_auto_deployment_pipeline(train_first=True)
    
    except KeyboardInterrupt:
        print(f"\nArrêt demandé par l'utilisateur")
    except Exception as e:
        print(f" Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()