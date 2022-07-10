"""
https://huggingface.co/transformers/v3.2.0/custom_datasets.html

https://huggingface.co/docs/transformers/model_doc/distilbert

"""
import pdb
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
from datasets import load_dataset
import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

from prepare_dataset import get_datasets


def clean_dataset(dataset):
    pass

def evaluate_pytorch_model(model, val_texts):
    predictions = []
    for txt in val_texts:
        encoding = tokenizer(txt, return_tensors="pt").to("cuda:0")
        logits = model(**encoding).logits
        predicted_class_id = logits.argmax().item()
        predictions.append(predicted_class_id)


    accuracy = accuracy_score(predictions, val_labels)
    recall = recall_score(predictions, val_labels)
    f1 = f1_score(predictions, val_labels)
    roc = roc_auc_score(predictions, val_labels)

    results_map = {}
    results_map['accuracy'] = accuracy
    results_map['recall'] = recall
    results_map['f1'] = f1
    results_map['roc_auc_score'] = roc
    print(results_map)


train_dataset, val_dataset,_,val_texts, train_labels,val_labels = get_datasets()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(device)


evaluate_pytorch_model(model, val_texts)

model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

n_epochs = 3
for epoch in range(n_epochs):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()

evaluate_pytorch_model(model, val_texts)