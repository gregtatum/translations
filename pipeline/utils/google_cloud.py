import json
import os
import shutil

from google.cloud import storage as RealStorage

storage: RealStorage

# Mock google.cloud.storage in tests, otherwise return the real thing.
if os.environ.get("PYTEST"):

    class MockedBlob:
        name: str
        bucket: "MockedBucket"

        def __init__(self, name, bucket) -> None:
            super().__init__()
            self.name = name
            self.bucket = bucket

        def download_to_filename(self, destination_path: str) -> None:
            if not os.environ.get("MOCKED_DOWNLOADS"):
                raise Exception(
                    "The mocked google cloud storage utility expected the MOCKED_DOWNLOADS environment variable to be set."
                )
            mocked_downloads = json.loads(os.environ.get("MOCKED_DOWNLOADS"))

            if not isinstance(mocked_downloads, dict):
                raise Exception(
                    "Expected the mocked downloads to be a json object mapping the URL to file path"
                )
            url = f"gs://{self.bucket.name}/{self.name}"
            source_file = mocked_downloads.get(url)
            if not source_file:
                print("[mocked gcs] MOCKED_DOWNLOADS:", mocked_downloads)
                raise Exception(f"Received a URL that was not in MOCKED_DOWNLOADS {url}")

            if not os.path.exists(source_file):
                raise Exception(f"The source file specified did not exist {source_file}")

            print("[mocked gcs] copying the file")
            print(f"[mocked gcs] from: {source_file}")
            print(f"[mocked gcs] to: {destination_path}")

            shutil.copyfile(source_file, destination_path)

    class MockedBucket:
        def __init__(self, name: str) -> None:
            self.name = name

        def blob(self, name: str):
            return MockedBlob(name, bucket=self)

    class MockedClient:
        @staticmethod
        def create_anonymous_client():
            return MockedClient()

        def bucket(self, bucket_name: str):
            return MockedBucket(bucket_name)

    class MockedStorage:
        Client = MockedClient

    storage = MockedStorage()
else:
    storage = RealStorage
