import os
import time
import tarfile
import boto3
import pandas as pd


REGION = "us-east-1"
BUCKET = "s3-credit-card-esgi"
ROLE_ARN = "arn:aws:iam::513816987855:role/SageMakerFraudRole"

IMAGE_URI = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"

ENDPOINT_NAME = "fraud-endpoint"

SUBNETS = [
    "subnet-0c1a790bb7651bcdf",
    "subnet-01e6579e04f42ef4c",
]

SECURITY_GROUPS = [
    "sg-05def7f4f02a29dca",
]


sagemaker = boto3.client("sagemaker", region_name=REGION)
runtime = boto3.client("sagemaker-runtime", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)


def wait_processing(job_name):
    print(f"Attente processing job: {job_name}")
    while True:
        desc = sagemaker.describe_processing_job(ProcessingJobName=job_name)
        status = desc["ProcessingJobStatus"]
        print("Processing status:", status)

        if status in ["Completed", "Failed", "Stopped"]:
            if status != "Completed":
                raise RuntimeError(desc.get("FailureReason", "Processing failed"))
            return

        time.sleep(30)


def wait_training(job_name):
    print(f"Attente training job: {job_name}")
    while True:
        desc = sagemaker.describe_training_job(TrainingJobName=job_name)
        status = desc["TrainingJobStatus"]
        print("Training status:", status)

        if status in ["Completed", "Failed", "Stopped"]:
            if status != "Completed":
                raise RuntimeError(desc.get("FailureReason", "Training failed"))
            return desc["ModelArtifacts"]["S3ModelArtifacts"]

        time.sleep(30)


def wait_endpoint(endpoint_name):
    print(f"Attente endpoint: {endpoint_name}")
    while True:
        desc = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        status = desc["EndpointStatus"]
        print("Endpoint status:", status)

        if status in ["InService", "Failed", "OutOfService"]:
            if status != "InService":
                raise RuntimeError(desc.get("FailureReason", "Endpoint failed"))
            return

        time.sleep(30)


def upload_file(local_path, s3_key):
    print(f"Upload {local_path} -> s3://{BUCKET}/{s3_key}")
    s3.upload_file(local_path, BUCKET, s3_key)
    return f"s3://{BUCKET}/{s3_key}"


def make_tar(local_file, tar_path, arcname):
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(local_file, arcname=arcname)


def run_etl():
    etl_s3_uri = upload_file("src/etl.py", "source/etl.py")

    job_name = "fraud-etl-processing-" + time.strftime("%Y-%m-%d-%H-%M-%S")

    sagemaker.create_processing_job(
        ProcessingJobName=job_name,
        RoleArn=ROLE_ARN,
        AppSpecification={
            "ImageUri": IMAGE_URI,
            "ContainerEntrypoint": [
                "python3",
                "/opt/ml/processing/input/code/etl.py",
            ],
            "ContainerArguments": [
                "--input-train", "/opt/ml/processing/input/train/fraudTrain.csv",
                "--input-test", "/opt/ml/processing/input/test/fraudTest.csv",
                "--output-dir", "/opt/ml/processing/output",
            ],
        },
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": "ml.t3.medium",
                "VolumeSizeInGB": 20,
            }
        },
        ProcessingInputs=[
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri": etl_s3_uri,
                    "LocalPath": "/opt/ml/processing/input/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "train",
                "S3Input": {
                    "S3Uri": f"s3://{BUCKET}/raw/fraudTrain.csv",
                    "LocalPath": "/opt/ml/processing/input/train",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "test",
                "S3Input": {
                    "S3Uri": f"s3://{BUCKET}/raw/fraudTest.csv",
                    "LocalPath": "/opt/ml/processing/input/test",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
        ],
        ProcessingOutputConfig={
            "Outputs": [
                {
                    "OutputName": "processed",
                    "S3Output": {
                        "S3Uri": f"s3://{BUCKET}/processed/",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ]
        },
    )

    wait_processing(job_name)


def run_training():
    os.makedirs("/tmp/sagemaker-train", exist_ok=True)
    train_tar = "/tmp/sagemaker-train/sourcedir.tar.gz"
    make_tar("src/train.py", train_tar, "train.py")

    source_s3_uri = upload_file(train_tar, "source/sourcedir.tar.gz")

    job_name = "fraud-training-" + time.strftime("%Y-%m-%d-%H-%M-%S")

    sagemaker.create_training_job(
        TrainingJobName=job_name,
        RoleArn=ROLE_ARN,
        AlgorithmSpecification={
            "TrainingImage": IMAGE_URI,
            "TrainingInputMode": "File",
        },
        HyperParameters={
            "sagemaker_program": "train.py",
            "sagemaker_submit_directory": source_s3_uri,
        },
        InputDataConfig=[
            {
                "ChannelName": "train",
                "ContentType": "text/csv",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{BUCKET}/processed/train.csv",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        ],
        OutputDataConfig={
            "S3OutputPath": f"s3://{BUCKET}/models/"
        },
        ResourceConfig={
            "InstanceType": "ml.m5.large",
            "InstanceCount": 1,
            "VolumeSizeInGB": 20,
        },
        StoppingCondition={
            "MaxRuntimeInSeconds": 3600
        },
    )

    model_data = wait_training(job_name)
    print("Model artifact:", model_data)
    return job_name, model_data


def create_or_update_endpoint(model_data):
    os.makedirs("/tmp/sagemaker-inference", exist_ok=True)
    inference_tar = "/tmp/sagemaker-inference/inference-source.tar.gz"
    make_tar("src/inference.py", inference_tar, "inference.py")

    inference_s3_uri = upload_file(inference_tar, "source/inference-source.tar.gz")

    suffix = time.strftime("%Y-%m-%d-%H-%M-%S")

    model_name = f"fraud-model-{suffix}"
    config_name = f"fraud-serverless-config-{suffix}"

    print("Création model:", model_name)

    sagemaker.create_model(
        ModelName=model_name,
        ExecutionRoleArn=ROLE_ARN,
        PrimaryContainer={
            "Image": IMAGE_URI,
            "ModelDataUrl": model_data,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": inference_s3_uri,
            },
        },
        VpcConfig={
            "Subnets": SUBNETS,
            "SecurityGroupIds": SECURITY_GROUPS,
        },
    )

    print("Création endpoint config:", config_name)

    sagemaker.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "ServerlessConfig": {
                    "MemorySizeInMB": 2048,
                    "MaxConcurrency": 2,
                },
            }
        ],
    )

    existing = sagemaker.list_endpoints(NameContains=ENDPOINT_NAME)["Endpoints"]

    if any(e["EndpointName"] == ENDPOINT_NAME for e in existing):
        print("Update endpoint existant")
        sagemaker.update_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=config_name,
        )
    else:
        print("Création nouvel endpoint")
        sagemaker.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=config_name,
        )

    wait_endpoint(ENDPOINT_NAME)


def test_endpoint():
    print("Téléchargement test.csv pour test endpoint")
    s3.download_file(BUCKET, "processed/test.csv", "/tmp/test.csv")

    df = pd.read_csv("/tmp/test.csv")
    X = df.drop(columns=["is_fraud"], errors="ignore")

    payload_path = "/tmp/payload_ci.csv"
    X.head(5).to_csv(payload_path, header=False, index=False)

    with open(payload_path, "rb") as f:
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Body=f.read(),
        )

    body = response["Body"].read().decode("utf-8")
    print("Réponse endpoint:")
    print(body)


def main():
    print("=== CI/CD ML START ===")

    run_etl()
    job_name, model_data = run_training()
    create_or_update_endpoint(model_data)
    test_endpoint()

    print("=== CI/CD ML FINI AVEC SUCCÈS ===")
    print("Training job:", job_name)
    print("Endpoint:", ENDPOINT_NAME)


if __name__ == "__main__":
    main()
