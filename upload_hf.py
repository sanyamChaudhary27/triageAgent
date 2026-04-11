import os
from huggingface_hub import HfApi, login
from dotenv import load_dotenv

load_dotenv()

def upload_to_hf():
    # Retrieve token from environment
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN not found in .env")
        return

    # API instance
    api = HfApi()
    
    # Login
    login(token=token)
    
    repo_id = "sanyamChaudhary27/customer-support-triage"
    local_dir = "."
    
    # Files to ignore during upload
    ignore_patterns = [".git*", ".env", "__pycache__", "*.pyc", "upload_hf.py", "temp_triage", "node_modules"]
    
    print(f"Uploading files to {repo_id}...")
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=ignore_patterns,
        delete_patterns=None # Only upload, don't delete remote files unless necessary
    )
    print("Upload complete!")

if __name__ == "__main__":
    upload_to_hf()
