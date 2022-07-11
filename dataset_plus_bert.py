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
from customModel import transfPlusEmbedModel
from config import params
import time

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset_data = get_datasets()

train_dataset = dataset_data['train_dataset']
val_dataset = dataset_data['val_dataset']
train_texts = dataset_data['train_texts']
val_texts = dataset_data['val_texts']
train_labels = dataset_data['train_labels']
val_labels = dataset_data['val_labels']
embedding_keys = dataset_data['embedding_keys']
author_to_ix = dataset_data['author_to_ix']
trainClaimAuthors = dataset_data['trainClaimAuthors']
valClaimAuthors = dataset_data['valClaimAuthors']
train_classes_count = dataset_data['train_classes_count']
#train_dataset, val_dataset,train_texts,val_texts, train_labels,val_labels, embeddingKeys, author_to_ix, trainClaimAuthors, valClaimAuthors = get_datasets()
class_balance = params['class_balance']

train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

model_file_name = params['model_name']+ "_" + str(time.time()) + ".pt"

def trainModel(model_name):

    best_accuracy = 0.0

    if model_name == "custom":
        p_model = transfPlusEmbedModel(embedding_keys, num_labels=2, 
        embedding_dim=params['claim_author_embedding_dim'],
        train_classes_count = train_classes_count,
        class_balance=class_balance )

    elif model_name == 'singleTransformer':
        p_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', 
        num_labels=2)

    p_model.to(device)

    optim = AdamW(p_model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    p_model.train()

    for epoch in range(params['n_epochs']):
            for batch in train_loader:
                optim.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                if model_name == 'custom':
                    claimAuthors = batch['claimAuthors'].to(device)
                    outputs = p_model(claimAuthorIX = claimAuthors,input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                elif model_name== 'singleTransformer':
                    outputs = p_model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs[0]
                loss.backward()
                optim.step()

            print("-"*20)
            print("epoch : ", epoch)
            p_model.eval()
            print("------------training set --------------")
            if model_name == "custom":
                evaluate_custom_model(p_model, author_to_ix, train_texts, train_labels, trainClaimAuthors)
                accuracy,_ = evaluate_custom_model(p_model, author_to_ix, val_texts, val_labels, valClaimAuthors)

            elif model_name== "singleTransformer":
                evaluate_pytorch_model(p_model, train_texts, train_labels)
                accuracy,_ = evaluate_pytorch_model(p_model, val_texts, val_labels)

            if accuracy > best_accuracy:
                print("saving model with name ", model_file_name)
                torch.save(p_model , "./models/"+ model_file_name)
                best_accuracy = accuracy
            else:
                break

            

trainModel(params['model_name'])
