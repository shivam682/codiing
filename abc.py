import os
import shutil
import time
from datetime import datetime, timedelta

def filter_recent_files(source_folder, dest_folder, days=14):
    """
    Copies JSON files modified in the last 'days' from source_folder to dest_folder.
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    cutoff_time = time.time() - days * 24 * 3600
    copied_files = []

    for file in os.listdir(source_folder):
        if file.endswith(".json"):
            file_path = os.path.join(source_folder, file)
            mtime = os.path.getmtime(file_path)
            if mtime >= cutoff_time:
                shutil.copy2(file_path, dest_folder)
                copied_files.append(file)

    print(f"Copied {len(copied_files)} files modified in last {days} days to '{dest_folder}'")
    return copied_files


if __name__ == "__main__":
    SOURCE_FOLDER = "all_crash_dumps"   # Your full dump archive
    TRAINING_FOLDER = "data_recent"     # Temporary folder for recent 14-day data

    # Step 1: Clear old training folder to avoid stale data
    if os.path.exists(TRAINING_FOLDER):
        shutil.rmtree(TRAINING_FOLDER)

    # Step 2: Copy last 14 days files
    recent_files = filter_recent_files(SOURCE_FOLDER, TRAINING_FOLDER, days=14)

    # Step 3: Trigger training on filtered data
    if recent_files:
        from fa_model_train import train_model
        print("Starting fine-tuning on last 14 days data...")
        train_model(TRAINING_FOLDER, num_classes=5, num_epochs=5, batch_size=4)
    else:
        print("No recent files found. Skipping training.")
# -----------------------------
# Step 3: Train Script + Eval with Fine-Tuning + Best Model Save
# -----------------------------

from sklearn.model_selection import train_test_split

def train_model(data_path, num_classes, num_epochs=5, batch_size=4, patience=2):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    dataset = CrashDataset(data_path, tokenizer)

    # Split into train and val sets
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    model = FAClassifier(num_classes=num_classes)

    model_path = "fa_classifier.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("🔁 Loaded existing model for fine-tuning.")
    else:
        print("🔁 No previous model found — training from scratch.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        for inputs, labels in train_loader:
            labels = torch.tensor(labels).to(device)
            outputs = model({k: v.to(device) for k, v in inputs.items()})
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()

        train_acc = correct / len(train_indices)
        print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Train Accuracy = {train_acc:.4f}")

        # Evaluate on validation set
        val_acc = evaluate_model(model, val_loader, dataset.idx2label, device, print_report=False)
        print(f"Epoch {epoch+1}: Validation Accuracy = {val_acc:.4f}")

        # Early stopping logic & best model save
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"💾 Saved best model with val acc: {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print("⏹ Early stopping triggered.")
            break

    print(f"Training finished. Best Validation Accuracy: {best_val_acc:.4f}")


def evaluate_model(model, dataloader, idx2label, device, print_report=True):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            labels = torch.tensor(labels).to(device)
            outputs = model({k: v.to(device) for k, v in inputs.items()})
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    if print_report:
        print(classification_report(y_true, y_pred, target_names=[idx2label[i] for i in range(len(idx2label))]))
    correct = sum(np.array(y_true) == np.array(y_pred))
    return correct / len(y_true)
# File: codebert_domain_adapt.py

import os
import json
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments

class LogTextDataset(Dataset):
    def __init__(self, json_folder, tokenizer, max_length=512):
        self.examples = []
        self.tokenizer = tokenizer
        for file in os.listdir(json_folder):
            if file.endswith(".json"):
                with open(os.path.join(json_folder, file), 'r') as f:
                    data = json.load(f)
                    all_logs = []

                    if data['type'] == 'firmware':
                        all_logs.extend([data.get('jira_summary', ''),
                                         data.get('wlan_fw_log', ''),
                                         data.get('callstack', '')])
                    else:
                        all_logs.extend([data.get('jira_summary', ''),
                                         data.get('dmesg', ''),
                                         data.get('wlan_driver_log', '')])

                    text = "\n".join([log.strip() for log in all_logs if log])
                    tokenized = tokenizer(text, truncation=True, padding='max_length', max_length=max_length)
                    self.examples.append(tokenized)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {key: torch.tensor(val) for key, val in self.examples[idx].items()}

def run_domain_adaptation(json_folder="data", output_dir="codebert-domain-adapted"):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModelForMaskedLM.from_pretrained("microsoft/codebert-base")

    dataset = LogTextDataset(json_folder, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=1,
        logging_steps=100,
        logging_dir=f"{output_dir}/logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("🔥 Starting domain adaptation training...")
    trainer.train()
    print(f"✅ Adapted model saved to {output_dir}")

if __name__ == "__main__":
    run_domain_adaptation()
