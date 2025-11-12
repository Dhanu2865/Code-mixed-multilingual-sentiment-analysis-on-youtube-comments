import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer
import pandas as pd

class CommentDataset(Dataset):
    def __init__(self, csv_path, tokenizer_name='xlm-roberta-base', max_length=128):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['comment_clean'])
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'sentiment_label': torch.tensor(row['sentiment_label'], dtype=torch.long),
            'toxicity_label': torch.tensor(row['toxicity_label'], dtype=torch.float),
            'anomaly_label': torch.tensor(row['anomaly_label'], dtype=torch.float),
        }

        # Optional: include language ID if you want cross-lingual weighting
        if 'language' in self.data.columns:
            item['language'] = row['language']
        return item


def get_dataloaders(train_path, val_path, test_path, batch_size=16, max_length=128):
    tokenizer_name = 'xlm-roberta-base'

    train_ds = CommentDataset(train_path, tokenizer_name, max_length)
    val_ds = CommentDataset(val_path, tokenizer_name, max_length)
    test_ds = CommentDataset(test_path, tokenizer_name, max_length)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, test_dl