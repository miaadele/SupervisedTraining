# Check that all required packages are installed.

import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
from sentence_transformers import SentenceTransformer

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Check that openpyxl is available (needed for reading .xlsx)
import openpyxl
print("openpyxl version:", openpyxl.__version__)

print("\nAll imports successful.")