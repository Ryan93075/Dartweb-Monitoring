import os
import re
import time
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from pymongo import MongoClient
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# ✅ GPU check
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)

# ✅ MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["local"]
collection = db["test_1"]

# ✅ Severity Mapping
severity_levels = ["None", "Low", "Medium", "High"]
severity_weights = {"None": 0.0, "Low": 1.0, "Medium": 2.0, "High": 3.0}

regex_patterns = {
    "emails": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "ip_addresses": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "crypto_wallets": r"\b(?:[13][a-km-zA-HJ-NP-Z1-9]{25,34}|bc1[a-zA-HJ-NP-Z0-9]{25,39})\b"
}

def clean_text(text):
    soup = BeautifulSoup(text, "html.parser")
    cleaned = soup.get_text()
    cleaned = re.sub(r"[^A-Za-z0-9\s]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()

def read_file(path):
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"⚠️ Error reading {path}: {e}")
            return None
    print(f"⚠️ Could not decode {path} with any encoding.")
    return None

def extract_entities(text):
    return {
        "emails": re.findall(regex_patterns["emails"], text),
        "ip_addresses": re.findall(regex_patterns["ip_addresses"], text),
        "crypto_wallets": re.findall(regex_patterns["crypto_wallets"], text),
    }

def keyword_density(text):
    words = text.lower().split()
    return len(words) / max(len(words), 1)

def compute_novelty(text, seen_hashes):
    content_hash = hash(text)
    if content_hash in seen_hashes:
        return 0.7
    else:
        seen_hashes.add(content_hash)
        return 1.5

def compute_score(severity, relevance, novelty, noise_factor):
    sev_weight = severity_weights.get(severity, 0.0)
    return round((relevance * sev_weight * novelty) / max(noise_factor, 0.1), 3)

def adjust_severity(severity, score):
    if severity == "High" and score < 1.0:
        return "Medium"
    if severity == "Medium" and score < 0.5:
        return "Low"
    return severity

def get_final_threat_level(severity, score):
    if severity == "High":
        return "Critical" if score >= 2.0 else "High (Low Confidence)"
    elif severity == "Medium":
        return "Medium" if score >= 1.5 else "Medium (Low Confidence)"
    else:
        return severity

# ✅ Dataset
class DarkWebDataset(Dataset):
    def __init__(self, texts, labels, types, tokenizer, max_len=512):
        self.texts, self.labels, self.types = texts, labels, types
        self.tokenizer, self.max_len = tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tokenizer(self.texts[i], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[i]),
            "type": self.types[i]
        }

# ✅ Train model ONCE and save
def train_and_save_model():
    df = pd.read_csv("darkweb_data_cleaned.csv")
    df["text"] = df["text"].apply(clean_text)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = DarkWebDataset(df["text"].tolist(), df["label"].tolist(), df["type"].tolist(), tokenizer)

    train_size = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=4, ignore_mismatched_sizes=True
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    best_val_acc = 0
    patience = 2
    wait = 0

    for epoch in range(5):
        model.train()
        total_loss = 0
        print(f"\n🔄 Epoch {epoch+1}/5")
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["label"].to(device)
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} Completed | Avg Training Loss: {avg_loss:.4f}")

        # ✅ Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device)
                )
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds.cpu() == batch["label"]).sum().item()
                total += batch["label"].size(0)

        val_acc = correct / total
        print(f"📊 Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            model.save_pretrained("fine_tuned_model")
            tokenizer.save_pretrained("fine_tuned_model")
            print("💾 Best model saved!")
        else:
            wait += 1
            if wait >= patience:
                print("⏹️ Early Stopping triggered!")
                break

# ✅ Scan folders
def scan_folders(base_dir, model, tokenizer, scanned, df_types):
    results = []
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if folder_path in scanned or not os.path.isdir(folder_path):
            continue

        highest_severity = "None"
        combined_text = ""
        max_conf = 0.0
        seen_hashes = set()

        for file in os.listdir(folder_path):
            if not file.endswith(".html"):
                continue

            content = read_file(os.path.join(folder_path, file))
            if not content:
                continue

            cleaned = clean_text(content)
            combined_text += cleaned + " "

            inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = F.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                conf = probs[0][pred].item()
                max_conf = max(max_conf, conf)

            if severity_levels.index(severity_levels[pred]) > severity_levels.index(highest_severity):
                highest_severity = severity_levels[pred]

        if highest_severity != "None" and combined_text.strip():
            novelty = compute_novelty(combined_text, seen_hashes)
            density = keyword_density(combined_text)
            noise_factor = 1 + max(0, len(combined_text.split()) / 1000 - density)
            relevance = max(0.4, 0.3 * density + 0.7 * max_conf)

            score = compute_score(highest_severity, relevance, novelty, noise_factor)
            adjusted_severity = adjust_severity(highest_severity, score)
            final_threat_level = get_final_threat_level(adjusted_severity, score)

            detected_type = "unknown"
            for t in df_types:
                if t.lower() in combined_text.lower():
                    detected_type = t
                    break

            results.append({
                "onionsite_url": folder,
                "severity": adjusted_severity,
                "final_threat_level": final_threat_level,
                "model_confidence": round(max_conf, 3),
                "violation": detected_type,
                "score": score,
                "entities": extract_entities(combined_text),
                "timestamp": datetime.utcnow(),
                "content_hash": hash(combined_text)
            })

            scanned.add(folder_path)

            print(f"✅ {folder} | Sev:{adjusted_severity} | Threat:{final_threat_level} | Conf:{round(max_conf,3)} | Type:{detected_type} | Score:{score}")

    return results

def main():
    if not os.path.exists("fine_tuned_model"):
        print("🚀 No fine-tuned model found. Training now...")
        train_and_save_model()

    tokenizer = BertTokenizer.from_pretrained("fine_tuned_model")
    model = BertForSequenceClassification.from_pretrained("fine_tuned_model").to(device)
    model.eval()

    df = pd.read_csv("darkweb_data_cleaned.csv")
    df_types = df["type"].unique()

    scanned = set()
    base_dir = "scan"

    while True:
        findings = scan_folders(base_dir, model, tokenizer, scanned, df_types)
        if findings:
            collection.insert_many(findings)
            print(f"✅ Inserted {len(findings)} new records.")
        else:
            print("🟢 No new threats.")
        time.sleep(5)

if __name__ == "__main__":
    main()
