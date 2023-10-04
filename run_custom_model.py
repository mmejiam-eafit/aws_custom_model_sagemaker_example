import sagemaker
import boto3
from time import gmtime, strftime, sleep

TIME_FORMAT = '%Y-%m-%d-%H-%M-%S'
SECONDS_IN_HOUR = 60*60
HOURS = 120

if __name__ == "__main__":
    sagemaker_session = sagemaker.Session()

    output_bucket = sagemaker.Session().default_bucket()
    output_prefix = "sagemaker/pruning_experiment_iterative"
    output_bucket_path = f"s3://{output_bucket}"

    # role = sagemaker.get_execution_role()
    region = boto3.Session().region_name
    job_name = f"pruning-experiment-iterative-{strftime(TIME_FORMAT, gmtime())}"

    training_image = "056057680849.dkr.ecr.us-east-1.amazonaws.com/chexnet-pruning-experiments:incremental"

    training_config = {
        "TrainingJobName": job_name,
        "AlgorithmSpecification": {
            "TrainingImage": training_image,
            "TrainingInputMode": "File"
        },
        "RoleArn": "arn:aws:iam::056057680849:role/aws_local_notebook",
        # "RoleArn": role,
        "OutputDataConfig": {"S3OutputPath": f"{output_bucket_path}/{output_prefix}/pruning_experiment_iterative"},
        "InputDataConfig": [
            {
                "ChannelName": "dataset",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": "s3://chexnet-dataset-divisions",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            },
            {
                "ChannelName": "database",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": "s3://chexnet-dataset-images",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            },
        ],
        "ResourceConfig": {
            "InstanceType": "ml.p3.2xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 92,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": HOURS * SECONDS_IN_HOUR
        }
    }

    # conn = boto3.client("s3")
    # print(conn.list_buckets())
    # contents = conn.list_objects(Bucket="chexnet-dataset-divisions", Prefix="")["Contents"]
    # print([f['Key'] for f in contents])

    client = boto3.client("sagemaker", region_name=region)
    client.create_training_job(**training_config)

    # while status != "Completed" and status != "Failed":
    #     sleep(60)
    status = client.describe_training_job(TrainingJobName=job_name)["TrainingJobStatus"]
    print(f"Training job with status {status}")
