
import os
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN")

HF_MODEL_REPO = os.environ.get(
    "HF_MODEL_REPO",
    "PallaviPatil0501/superkart-sales-model"
)

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN environment variable is not set."
    )

api = HfApi(token=HF_TOKEN)

api.create_repo(
    repo_id=HF_MODEL_REPO,
    repo_type="model",
    private=False,
    exist_ok=True
)

api.upload_file(
    path_or_fileobj="models/best_model.pkl",
    path_in_repo="best_model.pkl",
    repo_id=HF_MODEL_REPO,
    repo_type="model"
)

print(
    f"✅ Model registered: "
    f"https://huggingface.co/{HF_MODEL_REPO}"
)
