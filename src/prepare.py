# src/prepare.py
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os
from sklearn.datasets import load_iris

def prepare_data():
    print("--- Starting Data Preparation Stage ---")

    # 1. Загрузка параметров
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)

    # 2. Чтение данных
    try:
        # DVC гарантирует, что iris.csv будет здесь после 'dvc pull'
        df = pd.read_csv("data/raw/iris.csv")
    except FileNotFoundError:
         # Если данные не были предварительно сохранены в файле, используем sklearn для примера
        print("INFO: data/raw/iris.csv not found. Loading directly from sklearn.")
        iris = load_iris(as_frame=True)
        df = iris.frame
        df.columns = [c.replace(' (cm)', '').replace(' ', '_') for c in df.columns]

    # 3. Разделение признаков и целевой переменной
    target_col = params['prepare']['target_col']
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 4. Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=params['base']['test_size'], 
        random_state=params['base']['random_state']
    )

    # 5. Сохранение обработанных данных (OUTPUT для стадии 'prepare')
    os.makedirs("data/processed", exist_ok=True)

    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_frame(name=target_col).to_csv("data/processed/y_train.csv", index=False)
    y_test.to_frame(name=target_col).to_csv("data/processed/y_test.csv", index=False)

    print("Data preparation complete. Saved files to data/processed/")

if __name__ == "__main__":
    prepare_data()