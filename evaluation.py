import pdb
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from transformers import DistilBertTokenizerFast

#tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
from prepare_dataset import getTokenizer
import torch

tokenizer = getTokenizer()

def evaluateMistakes(predictions, val_text, val_labels, val_claim_authors=None):

    pdb.set_trace()
    mistakes = {}
    mistakes[0] = []
    mistakes[1] = []

    for row in range(len(predictions)):
        if predictions[row] != val_labels[row]:
            mistakes[val_labels[row]].append([val_text[row], val_claim_authors[row]]) 

    return mistakes

def getMetrics(predictions, val_labels):
    accuracy = accuracy_score(predictions, val_labels)
    recall = recall_score(predictions, val_labels)
    f1 = f1_score(predictions, val_labels)
    results_map = {}
    results_map['accuracy'] = accuracy
    results_map['recall'] = recall
    results_map['f1'] = f1
    print(results_map)
    return accuracy

def evaluate_pytorch_model(model, val_texts, val_labels):
    predictions = []
    for txt in val_texts:
        encoding = tokenizer(txt, return_tensors="pt").to("cuda:0")
        logits = model(**encoding).logits
        predicted_class_id = logits.argmax().item()
        predictions.append(predicted_class_id)

    return getMetrics(predictions, val_labels), predictions

def evaluate_custom_model(model, author_to_ix, val_texts, val_labels, val_claim_authors,device='cuda:0'):
    predictions = []
    count=0
    print(len(val_texts), len(val_labels), len(val_claim_authors))
    for txt in val_texts:

        if val_claim_authors[count] not in author_to_ix:
            author_ix = author_to_ix['unknown']
        else:
            author_ix = author_to_ix[val_claim_authors[count]]

        encoding = tokenizer(txt, return_tensors="pt").to(device)
        encoding['claimAuthorIX'] = torch.tensor(author_ix).to(device)
        logits = model(**encoding).logits
        predicted_class_id = logits.argmax().item()
        predictions.append(predicted_class_id)
        count+=1

    return getMetrics(predictions, val_labels), predictions
