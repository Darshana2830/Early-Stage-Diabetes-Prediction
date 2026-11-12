
import boto3
from botocore.exceptions import ClientError

REGION = "ap-southeast-2"
ENDPOINT = "early-diabetes-prediction-endpoint"

sm = boto3.client("sagemaker", region_name=REGION)

confirmation = input("Type DELETE to remove endpoint: ")

if confirmation == "DELETE":
    try:
        description = sm.describe_endpoint(EndpointName=ENDPOINT)
        config = description["EndpointConfigName"]

        sm.delete_endpoint(EndpointName=ENDPOINT)
        sm.get_waiter("endpoint_deleted").wait(
            EndpointName=ENDPOINT
        )
        print("Endpoint deleted successfully.")

        config_data = sm.describe_endpoint_config(
            EndpointConfigName=config
        )

        model_names = [
            item["ModelName"]
            for item in config_data["ProductionVariants"]
        ]

        sm.delete_endpoint_config(
            EndpointConfigName=config
        )

        for name in model_names:
            sm.delete_model(ModelName=name)

        print("Endpoint configuration and model resources deleted.")
        print("S3 model archive retained.")

    except ClientError as exc:
        print("AWS error:", exc)

else:
    print("Deletion cancelled.")
