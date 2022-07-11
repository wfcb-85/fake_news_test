from transformers import DistilBertForSequenceClassification
import torch
from transformers import DistilBertTokenizerFast
from config import params
from prepare_dataset import get_datasets
from evaluation import evaluate_custom_model, evaluateMistakes

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

model = torch.load('./models/modelscustom.pt').to('cpu')
model.eval()

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
valClaimAuthorsNames =  dataset_data['valClaimAuthorsNames'] 
#train_dataset, val_dataset,train_texts,val_texts, train_labels,val_labels, embeddingKeys, author_to_ix, trainClaimAuthors, valClaimAuthors = get_datasets()
class_balance = params['class_balance']

_, predictions = evaluate_custom_model(model, author_to_ix, val_texts, val_labels,
valClaimAuthors, device='cpu')

mistakes = evaluateMistakes(predictions, val_texts, val_labels, valClaimAuthorsNames)

print(mistakes)