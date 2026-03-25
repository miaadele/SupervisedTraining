# The data collator randomly masks 15% of tokens in each chunk
# The masked chunk is fed through BERT
# BERT predicts what the masked tokens should be
# The loss measures how wrong those predictions were
# The optimizer updates BERT’s weights to reduce the loss
# Repeat for every chunk, for several passes (epochs) through the data

import json
from pathlib import Path
import numpy as np
import torch

try:
    import accelerate
except ImportError:
    raise ImportError(
        "Please install accelerate with: pip install 'accelerate>=1.1.0'"
    )

from datasets import load_from_disk
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

# Device diagnostics. 
print("PyTorch version:", torch.__version__) #checks that PyTorch is installed correctly
print("CUDA available:", torch.cuda.is_available()) #checks that PyTorch can see the GPU
print("CUDA version:", torch.version.cuda) #verify where the model runs, either GPU or CPU

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    device_label = "cuda"
else:
    print("GPU: not available, using CPU")
    device_label = "cpu"


# Load tokenizer and datasets

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_path = Path("data") / "train_dataset"
eval_path = Path("data") / "eval_dataset"

if not train_path.exists() or not eval_path.exists():
    raise FileNotFoundError(
        "Could not find data/train_dataset or data/eval_dataset.\n"
        "Please run step4_prepare_data.py first."
    )

train_dataset = load_from_disk(str(train_path))
eval_dataset = load_from_disk(str(eval_path))

print(f"\nTraining chunks: {len(train_dataset)}")
print(f"Evaluation chunks: {len(eval_dataset)}")


# Load pretrained BERT with MLM head. 
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
print(f"Model parameters: {model.num_parameters():,}")


# Data collator to do the random masking automatically during training.
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)


# Training configuration
# eval_strategy="epoch" means evaluate once per epoch.
# save_strategy="epoch" means save a checkpoint once per epoch.

training_args = TrainingArguments(
    output_dir="./geneva_bert_checkpoints",

    # Core training settings
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,

    # Evaluation and saving
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,

    # Logging
    logging_steps=50,

    # Performance: this only matters if CUDA is available 
    #CUDA allows programs to use their GPU for general purpose computation
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=0,

    # Turn off external logging services
    report_to="none",
)

print("\nTraining configuration:")
print(f"  Device: {device_label}")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Train batch size: {training_args.per_device_train_batch_size}")
print(f"  Eval batch size: {training_args.per_device_eval_batch_size}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Warmup steps: {training_args.warmup_steps}")
print(f"  Mixed precision (fp16): {training_args.fp16}")

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Train
print("\n" + "=" * 60)
print("Starting fine-tuning...")
if torch.cuda.is_available():
    print("GPU detected: training should be relatively fast.")
else:
    print("No GPU detected: training will run on CPU and may be slower.")
print("Watch the training and evaluation loss decrease over time.")
print("=" * 60 + "\n")

train_result = trainer.train()

print("\n=== Training Complete ===")
print(f"Total training time: {train_result.metrics['train_runtime']:.1f} seconds")
print(f"Final training loss: {train_result.metrics['train_loss']:.4f}")


#Evaluate:
# Evaluation loss measures how well the model predicts masked words in text it did not train on
    #lower values indicate better predictions
# Perplexity measures how uncertain the model is when predicting the masked word/token

eval_results = trainer.evaluate()

print("\n=== Evaluation Results ===")
print(f"Eval loss: {eval_results['eval_loss']:.4f}")
print(f"Perplexity: {np.exp(eval_results['eval_loss']):.2f}")
print("(Lower perplexity means the model is less surprised by the text.)")


# Now we save the fine-tuned model
save_path = Path("geneva-bert")
save_path.mkdir(exist_ok=True)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"\nModel saved to: {save_path}/")
print("Files saved:")
for f in save_path.iterdir():
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"  {f.name} ({size_mb:.1f} MB)")


# and save training log for later analysis

Path("data").mkdir(exist_ok=True)

log_data = {
    "log_history": trainer.state.log_history,
    "train_runtime": train_result.metrics["train_runtime"],
    "train_loss": train_result.metrics["train_loss"],
    "eval_loss": eval_results["eval_loss"],
}

with open(Path("data") / "training_log.json", "w", encoding="utf-8") as f:
    json.dump(log_data, f, indent=2)

print("\nSaved training log to data/training_log.json")
print("Next step: compare baseline and fine-tuned MLM predictions.")