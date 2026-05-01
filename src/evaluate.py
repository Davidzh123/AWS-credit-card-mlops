import argparse
import json
import os
import tarfile

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def load_model(model_dir):
    model_tar_path = os.path.join(model_dir, "model.tar.gz")
    extracted_dir = os.path.join(model_dir, "extracted")

    os.makedirs(extracted_dir, exist_ok=True)

    with tarfile.open(model_tar_path, "r:gz") as tar:
        tar.extractall(path=extracted_dir)

    model_path = os.path.join(extracted_dir, "model.joblib")
    return joblib.load(model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="/opt/ml/processing/model")
    parser.add_argument("--test", type=str, default="/opt/ml/processing/test/test.csv")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/evaluation")

    args = parser.parse_args()

    print("Chargement modèle depuis:", args.model_dir)
    model = load_model(args.model_dir)

    print("Chargement test data:", args.test)
    df = pd.read_csv(args.test)

    X = df.drop(columns=["is_fraud"])
    y_true = df["is_fraud"].astype(int)

    y_pred = model.predict(X)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    evaluation = {
        "classification_metrics": {
            "accuracy": {"value": float(accuracy)},
            "precision": {"value": float(precision)},
            "recall": {"value": float(recall)},
            "f1": {"value": float(f1)},
        },
        "confusion_matrix": cm,
        "classification_report": report,
    }

    os.makedirs(args.output_dir, exist_ok=True)

    output_path = os.path.join(args.output_dir, "evaluation.json")

    with open(output_path, "w") as f:
        json.dump(evaluation, f, indent=2)

    print("Évaluation terminée.")
    print(json.dumps(evaluation["classification_metrics"], indent=2))
    print("Fichier écrit:", output_path)


if __name__ == "__main__":
    main()
