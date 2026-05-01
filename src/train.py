import argparse
import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="/opt/ml/input/data/train/train.csv")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=0)

    args = parser.parse_args()

    df = pd.read_csv(args.train)

    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"].astype(int)

    max_depth = None if args.max_depth == 0 else args.max_depth

    print("Hyperparamètres:")
    print("n_estimators =", args.n_estimators)
    print("max_depth =", max_depth)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y)

    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

    print("Modèle entraîné et sauvegardé.")


if __name__ == "__main__":
    main()