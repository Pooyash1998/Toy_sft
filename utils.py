# Had to add these stuff to avoid issue when i run on my mac
import os
import sys
import torch


def setup_env():
    if sys.platform == "darwin":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_torch_dtype(device):
    return torch.bfloat16 if device == "cuda" else torch.float32


def get_trainer_precision(device):
    return {"bf16": device == "cuda", "fp16": False}


def get_pin_memory(device):
    return device == "cuda"