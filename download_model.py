from pathlib import Path
from huggingface_hub import hf_hub_download

PROJECT_DIR = Path(__file__).parent
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_path = hf_hub_download(
    repo_id="andreolf/Qwen2.5-0.5B-Instruct-GGUF",
    filename="Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
    local_dir=str(MODEL_DIR),
)

print("\nModel downloaded successfully!")
print(f"Model location: {model_path}")
print("You can now run: streamlit run app.py")