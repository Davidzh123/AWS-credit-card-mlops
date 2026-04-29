import argparse
import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="/opt/ml/input/data/train/train.csv")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    args = parser.parse_args()

    df = pd.read_csv(args.train)

    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

    print("Modèle entraîné et sauvegardé.")


if __name__ == "__main__":
    main()
