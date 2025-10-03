# backend/app/utils/file_utils.py
import os

def get_file_path(upload_dir, filename):
    return os.path.join(upload_dir, filename)
