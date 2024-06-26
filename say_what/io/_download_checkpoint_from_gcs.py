"""
Download a checkpoint saved in google cloud storage

Last Modified: 04/15/2024
Author(s): Daniela Wiepert
"""

from pathlib import Path
from typing import Union
import os
from google.cloud import storage

def download_checkpoint_from_gcs(checkpoint_prefix: Union[str, Path], local_path: Union[str, Path],
                                 bucket):
    """
    Download a checkpoint from google cloud storage. Confirm authorization to do so with 'gcloud auth application-default login'

    :param checkpoint_prefix: str, location of checkpoint in bucket
    :param local_path: str, path to save checkpoint to on local computer
    :param bucket: gcs bucket
    """
    checkpoint_prefix = Path(checkpoint_prefix)
    local_path = Path(local_path).absolute()

    if not local_path.exists():
        os.makedirs(local_path)

    blobs = bucket.list_blobs(prefix=str(checkpoint_prefix))
    for b in blobs:
        new_path = local_path / os.path.basename(b.name)
        if not new_path.exists():
            b.download_to_filename(new_path)
            print(f'{new_path} download complete.')
    
    