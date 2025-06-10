# File: fa_model_train.py

import os
import json
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Optional

# Enable GPU optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# -----------------------------
# Step 1: Dataset Class
# -----------------------------

class CrashDataset(Dataset):
    def __init__(self, json_folder, tokenizer, max_length=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for file in os.listdir(json_folder):
            if file.endswith(".json"):
                with open(os.path.join(json_folder, file), 'r') as f:
                    data = json.load(f)
                    self.samples.append(data)

        self.label2idx = {label: idx for idx, label in enumerate(set(s['functional_area'] for s in self.samples if s.get('functional_area')))}
        self.idx2label = {v: k for k, v in self.label2idx.items()}

    def clean_text(self, text):
        return text[-4000:] if len(text) > 4000 else text

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        crash_type = sample.get('type', '')

        jira_summary = self.clean_text(sample.get('jira_summary', '') or '')
        inputs = {
            'jira': self.tokenizer(jira_summary, truncation=True, padding='max_length', max_length=self.max_length)
        }

        if crash_type == 'firmware':
            fw_log = self.clean_text(sample.get('wlan_fw_log', '') or '')
            callstack = self.clean_text(sample.get('callstack', '') or '')
            inputs['fw_log'] = self.tokenizer(fw_log, truncation=True, padding='max_length', max_length=self.max_length)
            inputs['callstack'] = self.tokenizer(callstack, truncation=True, padding='max_length', max_length=self.max_length)
        else:
            dmesg = self.clean_text(sample.get('dmesg', '') or '')
            driver = self.clean_text(sample.get('wlan_driver_log', '') or '')
            inputs['dmesg'] = self.tokenizer(dmesg, truncation=True, padding='max_length', max_length=self.max_length)
            inputs['driver'] = self.tokenizer(driver, truncation=True, padding='max_length', max_length=self.max_length)

        label = self.label2idx.get(sample.get('functional_area', ''), -1)
        return inputs, label

# -----------------------------
# Step 2: Model
# -----------------------------

class FAClassifier(nn.Module):
    def __init__(self, hidden_size=128, num_classes=5):
        super().__init__()
        self.bert = AutoModel.from_pretrained("microsoft/codebert-base")
        self.jira_fc = nn.Linear(768, hidden_size)
        self.fw_fc = nn.Linear(768, hidden_size)
        self.callstack_fc = nn.Linear(768, hidden_size)
        self.host_fc = nn.Linear(768 * 2, hidden_size)

        self.final_fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, num_classes)
        )

    def forward(self, inputs):
        jira_vec = self.bert(**inputs['jira']).pooler_output
        jira_vec = self.jira_fc(jira_vec)

        if 'fw_log' in inputs:
            fw_vec = self.bert(**inputs['fw_log']).pooler_output
            fw_vec = self.fw_fc(fw_vec)

            callstack_vec = self.bert(**inputs['callstack']).pooler_output
            callstack_vec = self.callstack_fc(callstack_vec)

            combined = torch.cat([jira_vec, fw_vec + callstack_vec], dim=1)

        else:
            dmesg_vec = self.bert(**inputs['dmesg']).pooler_output
            driver_vec = self.bert(**inputs['driver']).pooler_output

            host_vec = self.host_fc(torch.cat([dmesg_vec, driver_vec], dim=1))
            combined = torch.cat([jira_vec, host_vec], dim=1)

        return self.final_fc(combined)

# -----------------------------
# Step 3: Train Script + Eval
# -----------------------------

def evaluate_model(model, dataloader, idx2label, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            labels = torch.tensor(labels).to(device)
            outputs = model({k: v for k, v in inputs.items()})
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    print(classification_report(y_true, y_pred, target_names=[idx2label[i] for i in range(len(idx2label))]))

def train_model(data_path, num_classes, num_epochs=5, batch_size=4):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    full_dataset = CrashDataset(data_path, tokenizer)

    # Train/test split
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = FAClassifier(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        for inputs, labels in train_loader:
            labels = torch.tensor(labels).to(device)
            outputs = model({k: v for k, v in inputs.items()})
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()

        print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Train Accuracy = {correct / len(train_dataset):.4f}")
        print(f"Validation report for epoch {epoch+1}:")
        evaluate_model(model, val_loader, full_dataset.idx2label, device)

    torch.save(model.state_dict(), "fa_classifier.pt")
    print("Model saved as fa_classifier.pt")

# -----------------------------
# Step 4: Inference Script
# -----------------------------

def predict_functional_area(json_path, model_path="fa_classifier.pt"):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    with open(json_path, 'r') as f:
        sample = json.load(f)

    dataset = CrashDataset(os.path.dirname(json_path), tokenizer)
    model = FAClassifier(num_classes=len(dataset.label2idx))
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    inputs, _ = dataset[0]  # first and only sample
    with torch.no_grad():
        outputs = model({k: v.to(device) for k, v in inputs.items()})
        pred_idx = outputs.argmax(dim=1).item()
        print(f"Predicted Functional Area: {dataset.idx2label[pred_idx]}")

if __name__ == "__main__":
    train_model("data", num_classes=5)
    # To test inference on a single crash JSON file:
    # predict_functional_area("data/crash_123.json")
