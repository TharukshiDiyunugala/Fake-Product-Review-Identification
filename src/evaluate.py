from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
from sklearn.metrics import classification_report

def evaluate(model_path="../models/fake_review_model", data_path="../data/fake_reviews.csv"):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    df = pd.read_csv(data_path)
    texts = df['text_'].tolist()
    labels = df['label'].tolist()
    preds = []

    for txt in texts:
        inputs = tokenizer(txt, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        preds.append(torch.argmax(logits, dim=1).item())

    print(classification_report(labels, preds))
