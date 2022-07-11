import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers.modeling_outputs import TokenClassifierOutput
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from transformers import DistilBertForSequenceClassification

# Initializing a DistilBERT configuration
configuration = DistilBertConfig()

# Initializing a model from the configuration
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
Distilmodel = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

import torch.nn.functional as F

class CustomModule(nn.Module):
    def __init__(self, num_labels, vocab_size, embedding_dim):
        super(CustomModule,self).__init__()
        self.num_labels = num_labels

        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
        self.classifier = nn.Linear(768, num_labels)

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linearEmbs = nn.Linear(embedding_dim, 2)


    def forward(self, source=None, input_ids=None, attention_mask=None, labels=None):
        import pdb
        pdb.set_trace()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        #sequence_output = self.dropout(outputs[0])
        sequence_output = outputs[0]

        embeds = self.embeddings(torch.tensor(source)).view((1,-1))
        lin = F.relu(self.linearEmbs(embeds))
        out = sequence_output + lin 
        logits = F.log_softmax(out, dim=1)

        loss = None
        if labels is not None:

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions)

model = CustomModule(num_labels=2, vocab_size=10, embedding_dim=32)

model.eval()

from prepare_dataset import getTokenizer
tokenizer = getTokenizer()

text = "hi how are you"

encoding = tokenizer(text, return_tensors="pt")
encoding['source'] = 0

print("encoding :", encoding)


logits = model(**encoding).logits


text = "he was walking on the park while i was there but he talked to no one, man he is up into something"

encoding = tokenizer(text, return_tensors="pt")

print("encoding :", encoding)
encoding['source'] = 0

logits = model(**encoding).logits

predictions = torch.argmax(logits, dim=-1)

print(logits.view(-1, 2))

print(predictions)
