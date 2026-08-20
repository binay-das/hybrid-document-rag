import boto3
import logging

from botocore.exceptions import ClientError

from app.core.config import settings

logger = logging.getLogger(__name__)

class StorageService: 
    def __init__(self):
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name=settings.s3_region
        )
        self.bucket = settings.s3_bucket

    def ensure_bucket(self): 
        try: 
            self.client.head_bucket(Bucket = self.bucket)
            logger.info("Bucket exists, skipping creation")
        except ClientError: 
            self.client.create_bucket(Bucket = self.bucket)
            logger.info("Bucket created")

    def upload_file(self, file, key: str, content_type: str):
        self.client.upload_fileobj(
            file,
            self.bucket,
            key,
            ExtraArgs={"ContentType": content_type},
        )

    def get_file(self, key: str):
        return self.client.get_object(
            Bucket=self.bucket,
            Key=key,
        )["Body"]

    def delete_file(self, key: str):
        try:
            self.client.delete_object(
                Bucket=self.bucket,
                Key=key,
            )
        except ClientError as e:
            logger.error(f"Failed to delete file {key} from storage: {e}")