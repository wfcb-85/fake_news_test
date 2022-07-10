import torch
from datasets import load_dataset
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

dataset = load_dataset("datacommons_factcheck", "fctchk_politifact_wapo")

def get_lists_from_dataset(dataset):
    texts = []
    labels = []
    for i in dataset:
        texts.append(i[0])
        labels.append(i[1])
    return texts, labels

eval_dataset = dataset['train'][4000:]
train_dataset = dataset['train'][:4000]

cleaned_dataset = []

for row in dataset['train']:
    if row['review_rating'] in ['Pants on Fire', 'False', 'True']:
        if row['review_rating'] in ['Pants on Fire', 'False']:
            label=0
        else:
            label=1
        cleaned_dataset.append([row['claim_text'], label])

train_dataset = cleaned_dataset[:1500]
eval_dataset = cleaned_dataset[1500:]

train_texts, train_labels = get_lists_from_dataset(train_dataset)
val_texts, val_labels = get_lists_from_dataset(eval_dataset)

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)

def get_datasets():

    return train_dataset,val_dataset,train_texts,val_texts,train_labels, val_labels