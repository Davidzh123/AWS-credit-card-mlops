import argparse
import os
import pandas as pd
from sklearn.utils import resample


TARGET = "is_fraud"


def read_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, engine="python")


def basic_feature_engineering(df):
    df = df.copy()

    # Convertir la date de transaction en variables numériques simples
    if "trans_date_trans_time" in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(
            df["trans_date_trans_time"],
            errors="coerce"
        )
        df["trans_hour"] = df["trans_date_trans_time"].dt.hour.fillna(0).astype(int)
        df["trans_day"] = df["trans_date_trans_time"].dt.day.fillna(0).astype(int)
        df["trans_month"] = df["trans_date_trans_time"].dt.month.fillna(0).astype(int)

    # Garder seulement des colonnes utiles et légères
    keep_cols = [
        "amt",
        "lat",
        "long",
        "city_pop",
        "merch_lat",
        "merch_long",
        "trans_hour",
        "trans_day",
        "trans_month",
        "category",
        "gender",
        TARGET,
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # Gestion des valeurs manquantes
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("unknown")
        else:
            df[col] = df[col].fillna(df[col].median())

    # Encodage léger seulement sur category et gender
    categorical_cols = [c for c in ["category", "gender"] if c in df.columns]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df


def controlled_oversampling(df, target=TARGET, fraud_multiplier=3):
    majority = df[df[target] == 0]
    minority = df[df[target] == 1]

    print("Distribution avant oversampling:")
    print(df[target].value_counts())

    if len(minority) == 0:
        raise ValueError("Aucune fraude trouvée dans le dataset.")

    target_minority_size = min(len(majority), len(minority) * fraud_multiplier)

    minority_up = resample(
        minority,
        replace=True,
        n_samples=target_minority_size,
        random_state=42
    )

    result = pd.concat([majority, minority_up])
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Distribution après oversampling:")
    print(result[target].value_counts())

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-train", type=str, required=True)
    parser.add_argument("--input-test", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    print("Lecture train:", args.input_train)
    print("Lecture test:", args.input_test)

    train_df = read_csv_safe(args.input_train)
    test_df = read_csv_safe(args.input_test)

    print("Train brut:", train_df.shape)
    print("Test brut:", test_df.shape)

    train_processed = basic_feature_engineering(train_df)
    test_processed = basic_feature_engineering(test_df)

    # Aligner les colonnes train/test après encodage
    train_processed, test_processed = train_processed.align(
        test_processed,
        join="left",
        axis=1,
        fill_value=0
    )

    train_processed = controlled_oversampling(train_processed)

    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "train.csv")
    test_path = os.path.join(args.output_dir, "test.csv")

    train_processed.to_csv(train_path, index=False)
    test_processed.to_csv(test_path, index=False)

    print("ETL terminé.")
    print("Train final:", train_processed.shape)
    print("Test final:", test_processed.shape)
    print("Output train:", train_path)
    print("Output test:", test_path)


if __name__ == "__main__":
    main()