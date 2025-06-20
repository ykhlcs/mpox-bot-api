import os
import pandas as pd
from datasets import Dataset
from huggingface_hub import login
from dotenv import load_dotenv

# Load Hugging Face token
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

# Upload each dataset split individually
def upload_split(file_path, split_name, repo_id="aerynnnn/mpox-dataset"):
    print(f"📤 Uploading {split_name} split...")
    df = pd.read_csv(file_path)
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(repo_id, config_name=split_name)
    print(f"✅ {split_name} uploaded!")

# Paths and splits
uploads = {
    "data/faq.csv": "faq",
    "data/cleaned_who.csv": "who",
    "data/cleaned_cdc.csv": "cdc",
    "data/cleaned_followup.csv": "followup"
}

# Loop and upload
for path, name in uploads.items():
    upload_split(path, name)
