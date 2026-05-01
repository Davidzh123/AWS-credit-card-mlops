import boto3
import sagemaker

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import (
    ParameterString,
    ParameterFloat,
    ParameterInteger,
)
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.model_step import ModelStep

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.sklearn.model import SKLearnModel


REGION = "us-east-1"
BUCKET = "s3-credit-card-esgi"
ROLE_ARN = "arn:aws:iam::513816987855:role/SageMakerFraudRole"

PIPELINE_NAME = "fraud-sagemaker-ml-pipeline"
MODEL_PACKAGE_GROUP_NAME = "fraud-detection-model-group"

SUBNETS = [
    "subnet-0c1a790bb7651bcdf",
    "subnet-01e6579e04f42ef4c",
]

SECURITY_GROUPS = [
    "sg-05def7f4f02a29dca",
]


def get_pipeline():
    boto_session = boto3.Session(region_name=REGION)
    sagemaker_session = sagemaker.session.Session(
        boto_session=boto_session,
        default_bucket=BUCKET,
    )

    processing_instance_type = ParameterString(
        name="ProcessingInstanceType",
        default_value="ml.t3.medium",
    )

    training_instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value="ml.m5.large",
    )

    training_instance_count = ParameterInteger(
        name="TrainingInstanceCount",
        default_value=1,
    )

    min_f1_score = ParameterFloat(
        name="MinF1Score",
        default_value=0.80,
    )

    n_estimators = ParameterInteger(
        name="NEstimators",
        default_value=100,
    )

    max_depth = ParameterInteger(
        name="MaxDepth",
        default_value=0,
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

    processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=ROLE_ARN,
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name="fraud-pipeline-etl",
        sagemaker_session=sagemaker_session,
        network_config=sagemaker.network.NetworkConfig(
            subnets=SUBNETS,
            security_group_ids=SECURITY_GROUPS,
        ),
    )

    step_etl = ProcessingStep(
        name="ETLProcessData",
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
                output_name="train",
                source="/opt/ml/processing/output/train.csv",
                destination=processed_s3,
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/output/test.csv",
                destination=processed_s3,
            ),
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
        entry_point="train.py",
        source_dir="src",
        framework_version="1.2-1",
        py_version="py3",
        role=ROLE_ARN,
        instance_type=training_instance_type,
        instance_count=training_instance_count,
        base_job_name="fraud-pipeline-training",
        output_path=f"s3://{BUCKET}/pipeline/models",
        sagemaker_session=sagemaker_session,
        subnets=SUBNETS,
        security_group_ids=SECURITY_GROUPS,
        hyperparameters={
            "n-estimators": n_estimators,
            "max-depth": max_depth,
        },
    )

    step_train = TrainingStep(
        name="TrainFraudModel",
        estimator=estimator,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=step_etl.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            )
        },
    )

    evaluation_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=ROLE_ARN,
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name="fraud-pipeline-evaluation",
        sagemaker_session=sagemaker_session,
        network_config=sagemaker.network.NetworkConfig(
            subnets=SUBNETS,
            security_group_ids=SECURITY_GROUPS,
        ),
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    step_evaluate = ProcessingStep(
        name="EvaluateFraudModel",
        processor=evaluation_processor,
        code="src/evaluate.py",
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_etl.properties.ProcessingOutputConfig.Outputs[
                    "test"
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

    model = SKLearnModel(
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role=ROLE_ARN,
        entry_point="inference.py",
        source_dir="src",
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=sagemaker_session,
        vpc_config={
            "Subnets": SUBNETS,
            "SecurityGroupIds": SECURITY_GROUPS,
        },
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_evaluate.properties.ProcessingOutputConfig.Outputs[
                    "evaluation"
                ].S3Output.S3Uri
            ),
            content_type="application/json",
        )
    )

    register_args = model.register(
        content_types=["text/csv", "application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large", "ml.t2.medium"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    )

    step_register = ModelStep(
        name="RegisterFraudModelVersion",
        step_args=register_args,
    )

    condition_step = ConditionStep(
        name="CheckF1ScoreBeforeRegister",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=step_evaluate.name,
                    property_file=evaluation_report,
                    json_path="classification_metrics.f1.value",
                ),
                right=min_f1_score,
            )
        ],
        if_steps=[step_register],
        else_steps=[],
    )

    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[
            processing_instance_type,
            training_instance_type,
            training_instance_count,
            min_f1_score,
            n_estimators,
            max_depth,
            raw_train_s3,
            raw_test_s3,
            processed_s3,
            evaluation_s3,
        ],
        steps=[
            step_etl,
            step_train,
            step_evaluate,
            condition_step,
        ],
        sagemaker_session=sagemaker_session,
    )

    return pipeline


if __name__ == "__main__":
    pipeline = get_pipeline()

    print("Création ou mise à jour de la SageMaker Pipeline...")
    pipeline.upsert(role_arn=ROLE_ARN)

    print("Pipeline créée / mise à jour.")
    print("Nom:", PIPELINE_NAME)

    print("Lancement d'une exécution...")
    execution = pipeline.start(
        parameters={
            "NEstimators": 100,
            "MaxDepth": 0,
            "MinF1Score": 0.80,
        }
    )

    print("Execution ARN:")
    print(execution.arn)
