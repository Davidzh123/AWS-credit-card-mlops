import os
import json
import joblib
import pandas as pd


def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    return joblib.load(model_path)


def input_fn(request_body, request_content_type):
    if request_content_type == "text/csv":
        from io import StringIO
        return pd.read_csv(StringIO(request_body), header=None)

    if request_content_type == "application/json":
        data = json.loads(request_body)

        if isinstance(data, dict) and "instances" in data:
            return pd.DataFrame(data["instances"])

        if isinstance(data, list):
            return pd.DataFrame(data)

        return pd.DataFrame([data])

    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    predictions = model.predict(input_data)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[:, 1]
        return [
            {"prediction": int(pred), "fraud_probability": float(prob)}
            for pred, prob in zip(predictions, probabilities)
        ]

    return [{"prediction": int(pred)} for pred in predictions]


def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
