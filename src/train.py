# src/train.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import yaml
import mlflow
import os

# Используем логирование MLflow для отслеживания параметров и метрик
mlflow.set_experiment("DVC_MLflow_Pipeline_Iris")

def train_model():
    with mlflow.start_run() as run:
        print("--- Starting Model Training Stage ---")
        
        # 1. Загрузка параметров
        with open("params.yaml", 'r') as f:
            params = yaml.safe_load(f)
            
        train_params = params['train']
        base_params = params['base']

# 2. Загрузка обработанных данных (зависимость от data/processed)
        X_train = pd.read_csv("data/processed/X_train.csv")
        X_test = pd.read_csv("data/processed/X_test.csv")
        
        # Определяем название целевой колонки из params.yaml
        target_col = params['prepare']['target_col'] 
        
        # Чтение y_train и y_test: Читаем CSV и извлекаем единственную колонку по имени,
        # чтобы получить Pandas Series (как ожидается scikit-learn).
        y_train = pd.read_csv("data/processed/y_train.csv")[target_col]
        y_test = pd.read_csv("data/processed/y_test.csv")[target_col]
        
        # 3. Обучение модели
        model = LogisticRegression(
            solver=train_params['solver'],
            max_iter=train_params['max_iter'],
            random_state=base_params['random_state']
        )
        model.fit(X_train, y_train)
        
        # 4. Предсказание и метрика
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        
        # 5. Сохранение модели (OUTPUT для стадии 'train')
        model_path = "model.pkl"
        joblib.dump(model, model_path)
        print(f"Model trained. Accuracy: {acc:.4f}. Saved to {model_path}")
        
        # 6. MLflow Логирование (Шаг 3)
        mlflow.log_param("model_type", train_params['model_type'])
        mlflow.log_param("test_size", base_params['test_size'])
        mlflow.log_param("random_state", base_params['random_state'])
        mlflow.log_param("solver", train_params['solver'])
        
        mlflow.log_metric("accuracy", acc)
        
        # Логирование модели как артефакта MLflow
        mlflow.log_artifact(model_path) 
        
if __name__ == "__main__":
    train_model()