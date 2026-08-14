import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_NAME = "distilbert-base-uncased"
MODEL_DIR = "models/v2"
TRAIN_MODEL = True
BATCH_SIZE = 16
NUM_EPOCHS = 3

class SentimentDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None):
        self.texts = texts.tolist()
        self.labels = torch.tensor(labels.tolist(), dtype=torch.long) if labels is not None else None
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(MODEL_NAME)

        self.encodings = self.tokenizer(
            self.texts, 
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )     

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, index):
        item = {
            'input_ids': self.encodings['input_ids'][index], 
            'attention_mask': self.encodings['attention_mask'][index], 
        }

        if self.labels is not None:
            item['labels'] = self.labels[index]
        
        return item


def create_train_loader(train_val, tokenizer):
    train_texts = train_val['title'].fillna("").astype(str) + " " + train_val['sentence'].fillna("").astype(str)
    train_labels = train_val['score']

    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=False
    )

    return train_loader


def create_test_loader(test_val, tokenizer):
    test_texts = test_val['title'].fillna("").astype(str) + " " + test_val['sentence'].fillna("").astype(str)

    test_dataset = SentimentDataset(test_texts, tokenizer=tokenizer)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )

    return test_loader


def train(model, train_loader, optimizer):
    model.train()
    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch in tqdm(train_loader, total=len(train_loader)):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            optimizer.zero_grad()       # Resets the gradient
            outputs = model(**batch)    # Forward pass
            loss = outputs.loss         # Fetches the loss
            loss.backward()             # Backpropagation
            optimizer.step()            # Update the weights


def test(model, test_loader):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(**batch)    # Forward pass
            logits = outputs.logits     # Gets results
            pred = torch.argmax(logits, dim=-1) 
            results.append(pred.tolist())

    return results


def save_model(model, tokenizer):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)


def load_model_for_training():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    ).to(DEVICE)

    return model, tokenizer


def load_trained_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)

    return model, tokenizer


def main():
    if TRAIN_MODEL:
        model, tokenizer = load_model_for_training()
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

        train_val = pd.read_csv("data/train.csv")
        train_loader = create_train_loader(train_val, tokenizer)

        train(model, train_loader, optimizer)
        save_model(model, tokenizer)
    else:
        model, tokenizer = load_trained_model()

    test_val = pd.read_csv("data/test_no_score.csv")
    test_loader = create_test_loader(test_val, tokenizer)
    
    results = test(model, test_loader)

    with open("results/result_v2.txt", "w") as f:
        for val in np.concatenate(results):
            f.write(f"{val}\n")

if __name__ == "__main__":
    main()