"""
https://huggingface.co/transformers/v3.2.0/custom_datasets.html
https://huggingface.co/docs/transformers/model_doc/distilbert

"""
import torch
from transformers import DistilBertForSequenceClassification
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW

from prepare_dataset import get_datasets
from evaluation import evaluate_pytorch_model, evaluate_custom_model
from config import params
from customModel import transfPlusEmbedModel
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_dataset, val_dataset,train_texts,val_texts, train_labels,val_labels, embeddingKeys, author_to_ix, trainClaimAuthors, valClaimAuthors = get_datasets()

customModel = transfPlusEmbedModel(embeddingKeys, 2, 32 )
customModel.to(device)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.to(device)

model.train()

train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

optim = AdamW(model.parameters(), lr=params['lr'])

def trainCustomModel(p_model):
    optim = AdamW(p_model.parameters(), lr=params['lr'])
    p_model.train()

    for epoch in range(params['n_epochs']):
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            claimAuthors = batch['claimAuthors'].to(device)
            outputs = p_model(claimAuthorIX = claimAuthors,input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optim.step()
        print("-"*20)
        print("epoch : ", epoch)
        p_model.eval()
        print("------------training set --------------")
        evaluate_custom_model(p_model, author_to_ix, train_texts, train_labels, trainClaimAuthors)
        print("-"*20)
        print("------------test set --------------")
        evaluate_custom_model(p_model, author_to_ix, val_texts, val_labels, valClaimAuthors)
        print("-"*20)
        p_model.train()

trainCustomModel(customModel)
raise ValueError("Dsa")

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

    print("-"*20)
    print("epoch : ", epoch)
    model.eval()
    print("------------training set --------------")
    evaluate_pytorch_model(model, train_texts, train_labels)
    print("-"*20)
    print("------------test set --------------")
    evaluate_pytorch_model(model, val_texts, val_labels)
    print("-"*20)
    model.train()

model.eval()

