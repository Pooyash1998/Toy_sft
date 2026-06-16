import argparse
import json
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import setup_env

setup_env()

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

BASE_MODEL_ID = cfg["model"]["base_model_id"]
MODEL_PATH    = cfg["model"]["model_path"]
OUTPUT_DIR    = cfg["data"]["output_dir"]
MERGED_DIR    = cfg["data"].get("merged_dir", "./merged_model")

# Load on CPU for merge to avoid GPU OOM
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    trust_remote_code=True,
    device_map="cpu",
    cache_dir=MODEL_PATH,
)
model  = PeftModel.from_pretrained(base, OUTPUT_DIR)
merged = model.merge_and_unload()
merged.save_pretrained(MERGED_DIR)

tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(MERGED_DIR)

# Patch config.json for MLX compatibility (rope_theta must be top-level :(( )
config_path = f"{MERGED_DIR}/config.json"
with open(config_path) as f:
    config = json.load(f)
rope_params = config.get("rope_parameters", {})
if "rope_theta" in rope_params and "rope_theta" not in config:
    config["rope_theta"] = rope_params["rope_theta"]
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

print(f"Merged model saved to {MERGED_DIR}")