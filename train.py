# clean_and_train_sbert.py

import json
import os
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
os.environ["WANDB_DISABLED"] = "true"


# ----------------------------
# Step 1: Load + Clean Dataset
# ----------------------------
json_file = 'dataset.json'   # <-- dataset name

with open(json_file, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

clean_data = []

for item in raw_data:
    q = item.get("input")
    a = item.get("output")

    # Remove invalid rows
    if q is None or a is None:
        continue
    if not isinstance(q, str) or not isinstance(a, str):
        continue
    if q.strip() == "" or a.strip() == "":
        continue

    clean_data.append({"input": q, "output": a})

print(f"🔥 Clean data size: {len(clean_data)} (removed {len(raw_data)-len(clean_data)} invalid rows)")


# ----------------------------
# Step 2: Convert to SBERT Training Examples
# ----------------------------
train_examples = [
    InputExample(texts=[item["input"], item["output"]], label=1.0)
    for item in clean_data
]

print("📌 Training examples prepared.")


# ----------------------------
# Step 3: Load Pretrained SBERT
# ----------------------------
print("🤖 Loading SBERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')


# ----------------------------
# Step 4: Training Setup
# ----------------------------
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

epochs = 2
warmup_steps = int(len(train_dataloader) * epochs * 0.1)

print("🔥 Training started...")


# ----------------------------
# Step 5: Fine-Tune Model
# ----------------------------
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=epochs,
    warmup_steps=warmup_steps,
    show_progress_bar=True
)


# ----------------------------
# Step 6: Save Fine-Tuned SBERT
# ----------------------------
output_dir = "fine_tuned_sbert"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model.save(output_dir)

print("✅ Fine-tuning complete!")
print(f"🎉 Model saved to: {output_dir}")