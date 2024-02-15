import io
import json
import os
import shutil

import requests

from pipeline.common.logging import get_logger

logger = get_logger(__file__)


def stream_download_to_file(url: str, destination: str) -> None:
    """
    Streams a download to a file using 1mb chunks.
    """
    response = requests.get(url, stream=True)
    if not response.ok:
        raise Exception(f"Unable to download file from {url}")
    with open(destination, "wb") as f:
        logger.info("Streaming downloading: {url}")
        logger.info("To: {destination}")
        # Stream to disk in 1 megabyte chunks.
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
