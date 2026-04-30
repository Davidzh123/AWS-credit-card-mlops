import time
import tarfile
import boto3
import os

REGION = "us-east-1"
BUCKET = "s3-credit-card-esgi"
ROLE_ARN = "arn:aws:iam::513816987855:role/SageMakerFraudRole"

ENDPOINT_NAME = "fraud-endpoint"

SUBNETS = [
    "subnet-0c1a790bb7651bcdf",
    "subnet-01e6579e04f42ef4c",
]

SECURITY_GROUPS = [
    "sg-05def7f4f02a29dca",
]

INSTANCE_TYPE = "ml.t2.medium"
INSTANCE_COUNT = 1

sm = boto3.client("sagemaker", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)


def package_inference():
    """
    Package src/inference.py dans un fichier tar.gz
    puis l'upload dans S3.
    """
    tar_path = "/tmp/inference-source.tar.gz"

    if not os.path.exists("src/inference.py"):
        raise FileNotFoundError("src/inference.py introuvable")

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add("src/inference.py", arcname="inference.py")

    s3_key = "source/inference-source.tar.gz"
    s3.upload_file(tar_path, BUCKET, s3_key)

    return f"s3://{BUCKET}/{s3_key}"


def get_latest_completed_training_job():
    """
    Récupère le dernier training job SageMaker terminé.
    """
    response = sm.list_training_jobs(
        NameContains="fraud-training",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=10,
    )

    for job in response["TrainingJobSummaries"]:
        if job["TrainingJobStatus"] == "Completed":
            return job["TrainingJobName"]

    raise Exception("Aucun training job Completed trouvé.")


def endpoint_exists(endpoint_name):
    """
    Vérifie si l'endpoint existe déjà.
    """
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        return True
    except sm.exceptions.ClientError:
        return False


def wait_for_endpoint(endpoint_name):
    """
    Attend que l'endpoint soit InService ou Failed.
    """
    print("Attente du déploiement de l'endpoint...")

    while True:
        response = sm.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]

        print("Endpoint status:", status)

        if status == "InService":
            print("Endpoint prêt.")
            return

        if status == "Failed":
            reason = response.get("FailureReason", "Raison inconnue")
            raise Exception(f"Endpoint FAILED : {reason}")

        time.sleep(30)


def main():
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")

    print("Packaging inference.py...")
    inference_s3_uri = package_inference()
    print("Inference source:", inference_s3_uri)

    training_job_name = get_latest_completed_training_job()
    print("Training job utilisé:", training_job_name)

    training = sm.describe_training_job(
        TrainingJobName=training_job_name
    )

    model_data = training["ModelArtifacts"]["S3ModelArtifacts"]
    image_uri = training["AlgorithmSpecification"]["TrainingImage"]

    print("Model data:", model_data)
    print("Image URI:", image_uri)

    model_name = f"fraud-model-vpc-cicd-{timestamp}"
    config_name = f"fraud-endpoint-config-vpc-cicd-{timestamp}"

    print("Création du model:", model_name)

    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=ROLE_ARN,
        PrimaryContainer={
            "Image": image_uri,
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

    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": INSTANCE_COUNT,
                "InstanceType": INSTANCE_TYPE,
            }
        ],
    )

    if endpoint_exists(ENDPOINT_NAME):
        print("Endpoint existe déjà, update...")
        sm.update_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=config_name,
        )
    else:
        print("Création endpoint...")
        sm.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=config_name,
        )

    print("Déploiement lancé.")
    print("Endpoint:", ENDPOINT_NAME)
    print("Model:", model_name)
    print("Endpoint config:", config_name)

    wait_for_endpoint(ENDPOINT_NAME)


if __name__ == "__main__":
    main()