"""
https://huggingface.co/transformers/v3.2.0/custom_datasets.html
https://huggingface.co/docs/transformers/model_doc/distilbert

"""
import torch
from transformers import DistilBertForSequenceClassification
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW

from prepare_dataset import get_datasets
from evaluation import evaluate_pytorch_model

from config import params

train_dataset, val_dataset,_,val_texts, train_labels,val_labels = get_datasets()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(device)

model.train()

train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

optim = AdamW(model.parameters(), lr=params['lr'])

for epoch in range(params['n_epochs']):
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

evaluate_pytorch_model(model, val_texts, val_labels)