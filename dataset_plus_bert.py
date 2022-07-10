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

def get_lists_from_dataset(dataset):
    texts = []
    labels = []
    for i in dataset:
        texts.append(i[0])
        labels.append(i[1])
    return texts, labels

dataset = load_dataset("datacommons_factcheck", "fctchk_politifact_wapo")

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


"""
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()
"""

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


"""
input = tokenizer("hello my dog", return_tensors="pt").to("cuda:0")
logits = model(**input).logits
predicted_class_id = logits.argmax().item()
"""


evaluate_pytorch_model(model, val_texts)


#predictions = trainer.predict(val_dataset)