import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm
from src.data import load_data, split_data
from src.model import load_model

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts.iloc[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        item['labels'] = torch.tensor(int(self.labels.iloc[idx]))
        return item

def train():
    df = load_data()
    train_texts, test_texts, train_labels, test_labels = split_data(df)
    tokenizer, model = load_model()
    train_dataset = ReviewDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    model.train()

    for epoch in range(2):  # train for 2 epochs
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            outputs = model(**{k: v for k, v in batch.items()})
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())

    model.save_pretrained("../models/fake_review_model")
    tokenizer.save_pretrained("../models/fake_review_tokenizer")

if __name__ == "__main__":
    train()
