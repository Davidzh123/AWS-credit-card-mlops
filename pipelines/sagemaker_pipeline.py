import os
import json
import tarfile
import boto3
import sagemaker

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString, ParameterFloat, ParameterInteger
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ScriptProcessor
from sagemaker.network import NetworkConfig


REGION = "us-east-1"
BUCKET = "s3-credit-card-esgi"
ROLE_NAME = "SageMakerFraudRole"

PIPELINE_NAME = "fraud-ml-pipeline"
MODEL_PACKAGE_GROUP_NAME = "fraud-credit-card-model-group"

SUBNETS = [
    "subnet-0c1a790bb7651bcdf",
    "subnet-01e6579e04f42ef4c",
]

SECURITY_GROUPS = [
    "sg-05def7f4f02a29dca",
]


def write_evaluation_script():
    os.makedirs("src", exist_ok=True)

    code = r'''
import os
import json
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


MODEL_TAR = "/opt/ml/processing/model/model.tar.gz"
MODEL_DIR = "/opt/ml/processing/model/extracted"
TEST_PATH = "/opt/ml/processing/test/test.csv"
OUTPUT_DIR = "/opt/ml/processing/evaluation"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with tarfile.open(MODEL_TAR, "r:gz") as tar:
        tar.extractall(MODEL_DIR)

    model_path = os.path.join(MODEL_DIR, "model.joblib")
    model = joblib.load(model_path)

    df = pd.read_csv(TEST_PATH)

    if "is_fraud" not in df.columns:
        raise ValueError("La colonne is_fraud est absente du fichier test.csv")

    X_test = df.drop(columns=["is_fraud"])
    y_test = df["is_fraud"].astype(int)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    evaluation = {
        "binary_classification_metrics": {
            "accuracy": {"value": float(accuracy)},
            "precision": {"value": float(precision)},
            "recall": {"value": float(recall)},
            "f1": {"value": float(f1)},
        },
        "confusion_matrix": cm,
        "classification_report": report,
    }

    output_path = os.path.join(OUTPUT_DIR, "evaluation.json")

    with open(output_path, "w") as f:
        json.dump(evaluation, f, indent=4)

    print("Evaluation terminée")
    print(json.dumps(evaluation, indent=4))


if __name__ == "__main__":
    main()
'''

    with open("src/evaluate_pipeline.py", "w") as f:
        f.write(code)


def get_role_arn():
    boto_session = boto3.Session(region_name=REGION)
    account_id = boto_session.client("sts").get_caller_identity()["Account"]
    return f"arn:aws:iam::{account_id}:role/{ROLE_NAME}"


def get_pipeline():
    write_evaluation_script()

    boto_session = boto3.Session(region_name=REGION)

    pipeline_session = PipelineSession(
        boto_session=boto_session,
        default_bucket=BUCKET,
    )

    role_arn = get_role_arn()

    processing_instance_type = ParameterString(
        name="ProcessingInstanceType",
        default_value="ml.t3.medium",
    )

    training_instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value="ml.m5.large",
    )

    min_f1_score = ParameterFloat(
        name="MinF1Score",
        default_value=0.80,
    )

    raw_train_s3 = ParameterString(
        name="RawTrainData",
        default_value=f"s3://{BUCKET}/raw/fraudTrain.csv",
    )

    raw_test_s3 = ParameterString(
        name="RawTestData",
        default_value=f"s3://{BUCKET}/raw/fraudTest.csv",
    )

    processed_s3 = ParameterString(
        name="ProcessedDataS3Uri",
        default_value=f"s3://{BUCKET}/pipeline/processed",
    )

    evaluation_s3 = ParameterString(
        name="EvaluationS3Uri",
        default_value=f"s3://{BUCKET}/pipeline/evaluation",
    )

    network_config = NetworkConfig(
        subnets=SUBNETS,
        security_group_ids=SECURITY_GROUPS,
    )

    processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role_arn,
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name="fraud-pipeline-etl",
        sagemaker_session=pipeline_session,
        network_config=network_config,
    )

    step_etl = ProcessingStep(
        name="FraudETL",
        processor=processor,
        code="src/etl.py",
        inputs=[
            ProcessingInput(
                source=raw_train_s3,
                destination="/opt/ml/processing/input/train",
            ),
            ProcessingInput(
                source=raw_test_s3,
                destination="/opt/ml/processing/input/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="processed_data",
                source="/opt/ml/processing/output",
                destination=processed_s3,
            )
        ],
        job_arguments=[
            "--input-train",
            "/opt/ml/processing/input/train/fraudTrain.csv",
            "--input-test",
            "/opt/ml/processing/input/test/fraudTest.csv",
            "--output-dir",
            "/opt/ml/processing/output",
        ],
    )

    estimator = SKLearn(
        entry_point="src/train.py",
        framework_version="1.2-1",
        role=role_arn,
        instance_type=training_instance_type,
        instance_count=1,
        base_job_name="fraud-pipeline-training",
        sagemaker_session=pipeline_session,
        output_path=f"s3://{BUCKET}/pipeline/models",
        subnets=SUBNETS,
        security_group_ids=SECURITY_GROUPS,
    )

    step_train = TrainingStep(
        name="FraudTrainModel",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_etl.properties.ProcessingOutputConfig.Outputs[
                    "processed_data"
                ].S3Output.S3Uri,
                content_type="text/csv",
            )
        },
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    evaluator = SKLearnProcessor(
        framework_version="1.2-1",
        role=role_arn,
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name="fraud-pipeline-evaluation",
        sagemaker_session=pipeline_session,
        network_config=network_config,
    )

    step_evaluate = ProcessingStep(
        name="FraudEvaluateModel",
        processor=evaluator,
        code="src/evaluate_pipeline.py",
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_etl.properties.ProcessingOutputConfig.Outputs[
                    "processed_data"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=evaluation_s3,
            )
        ],
        property_files=[evaluation_report],
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=f"{evaluation_s3}/evaluation.json",
            content_type="application/json",
        )
    )

    step_register = RegisterModel(
        name="FraudRegisterModel",
        estimator=estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv", "application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large", "ml.t2.medium"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    )

    condition_f1 = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_evaluate.name,
            property_file=evaluation_report,
            json_path="binary_classification_metrics.f1.value",
        ),
        right=min_f1_score,
    )

    step_condition = ConditionStep(
        name="FraudCheckF1BeforeRegister",
        conditions=[condition_f1],
        if_steps=[step_register],
        else_steps=[],
    )

    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[
            processing_instance_type,
            training_instance_type,
            min_f1_score,
            raw_train_s3,
            raw_test_s3,
            processed_s3,
            evaluation_s3,
        ],
        steps=[
            step_etl,
            step_train,
            step_evaluate,
            step_condition,
        ],
        sagemaker_session=pipeline_session,
    )

    return pipeline


if __name__ == "__main__":
    pipeline = get_pipeline()

    print("Création ou mise à jour de la pipeline SageMaker...")
    pipeline.upsert(role_arn=get_role_arn())

    print("Pipeline créée / mise à jour.")
    print("Nom:", PIPELINE_NAME)

    print("Lancement de la pipeline...")
    execution = pipeline.start()

    print("Execution ARN:")
    print(execution.arn)