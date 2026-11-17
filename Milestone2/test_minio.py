
from minio import Minio

client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="ChangeThisPassword123!",
    secure=False
)

print("Connected:", client.bucket_exists("models"))
