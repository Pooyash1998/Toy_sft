import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

BASE_MODEL_ID = cfg["model"]["base_model_id"]
MODEL_PATH = cfg["model"]["model_path"]
OUTPUT_DIR = cfg["data"]["output_dir"]

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    dtype=torch.float16,
    trust_remote_code=True,
    cache_dir=MODEL_PATH,
)
model = PeftModel.from_pretrained(base, OUTPUT_DIR)
merged = model.merge_and_unload()
merged.save_pretrained("./merged_model")

tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained("./merged_model")

# MLX requires rope_theta at top level; Granite nests it inside rope_parameters
config_path = "./merged_model/config.json"
with open(config_path) as f:
    config = json.load(f)
rope_params = config.get("rope_parameters", {})
if "rope_theta" in rope_params and "rope_theta" not in config:
    config["rope_theta"] = rope_params["rope_theta"]
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

print("Merged and saved to ./merged_model")
