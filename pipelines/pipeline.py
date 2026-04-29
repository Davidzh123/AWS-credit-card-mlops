import sagemaker
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput


REGION = "us-east-1"
BUCKET = "s3-credit-card-esgi"
ROLE_NAME = "SageMakerFraudRole"


def main():
    boto_session = sagemaker.session.Session().boto_session
    account_id = boto_session.client("sts").get_caller_identity()["Account"]

    role = f"arn:aws:iam::{account_id}:role/{ROLE_NAME}"

    sagemaker_session = sagemaker.Session(
        boto_session=boto_session,
        default_bucket=BUCKET
    )

    processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        base_job_name="fraud-etl-processing",
        sagemaker_session=sagemaker_session,
    )

    processor.run(
        code="src/etl.py",
        inputs=[
            ProcessingInput(
                source=f"s3://{BUCKET}/raw/fraudTrain.csv",
                destination="/opt/ml/processing/input/fraudTrain.csv",
            ),
            ProcessingInput(
                source=f"s3://{BUCKET}/raw/fraudTest.csv",
                destination="/opt/ml/processing/input/fraudTest.csv",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=f"s3://{BUCKET}/processed/",
            )
        ],
        arguments=[
            "--input-train", "/opt/ml/processing/input/fraudTrain.csv",
            "--input-test", "/opt/ml/processing/input/fraudTest.csv",
            "--output-dir", "/opt/ml/processing/output",
        ],
    )

    print("ETL SageMaker terminé.")
    print(f"Résultats disponibles dans : s3://{BUCKET}/processed/")


if __name__ == "__main__":
    main()
