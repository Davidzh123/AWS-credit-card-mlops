import argparse
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-train", type=str, default="/opt/ml/processing/input/fraudTrain.csv")
    parser.add_argument("--input-test", type=str, default="/opt/ml/processing/input/fraudTest.csv")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    args = parser.parse_args()

    train_df = pd.read_csv(args.input_train)
    test_df = pd.read_csv(args.input_test)

    df = pd.concat([train_df, test_df], ignore_index=True)

    columns_to_drop = [
        "Unnamed: 0",
        "cc_num",
        "first",
        "last",
        "street",
        "trans_num",
        "trans_date_trans_time"
    ]

    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

    # Gestion des valeurs manquantes
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("unknown")
        else:
            df[col] = df[col].fillna(df[col].mean())

    # Feature engineering date de naissance
    if "dob" in df.columns:
        df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
        df["birth_year"] = df["dob"].dt.year
        df["birth_year"] = df["birth_year"].fillna(df["birth_year"].median())
        df = df.drop(columns=["dob"])

    target = "is_fraud"

    if target not in df.columns:
        raise ValueError(f"Colonne cible absente : {target}")

    # Encodage catégoriel
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df.drop(columns=[target])
    y = df[target]

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Oversampling SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    processed_df = pd.DataFrame(X_resampled, columns=X.columns)
    processed_df[target] = y_resampled

    train_data, test_data = train_test_split(
        processed_df,
        test_size=0.2,
        random_state=42,
        stratify=processed_df[target]
    )

    os.makedirs(args.output_dir, exist_ok=True)

    train_data.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    test_data.to_csv(os.path.join(args.output_dir, "test.csv"), index=False)

    print("ETL terminé.")
    print("Train shape:", train_data.shape)
    print("Test shape:", test_data.shape)


if __name__ == "__main__":
    main()
