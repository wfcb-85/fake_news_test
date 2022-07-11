import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification
from transformers.modeling_outputs import TokenClassifierOutput

class transfPlusEmbedModel(nn.Module):

    def __init__(self, embeddingKeys, num_labels, embedding_dim, train_classes_count,class_balance):
        super(transfPlusEmbedModel,self).__init__()
        self.class_weights = torch.FloatTensor([0.5,0.5])
        if class_balance:
            _sum = train_classes_count[0] + train_classes_count[1]
            zero_weight = 1-(train_classes_count[0]/_sum)
            one_weight = 1-(train_classes_count[1]/_sum)
            self.class_weights = torch.FloatTensor([zero_weight, one_weight])
            print("balancing ", self.class_weights)

        self.num_labels = num_labels

        self.embeddingKeys = embeddingKeys

        self.vocab_size = len(self.embeddingKeys)

        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
        self.classifier = nn.Linear(768, num_labels)

        self.embeddings = nn.Embedding(len(embeddingKeys), embedding_dim)
        self.linearEmbs = nn.Linear(embedding_dim, 2)

        self.loss = nn.CrossEntropyLoss(self.class_weights)


    def forward(self, claimAuthorIX=None, input_ids=None, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        #sequence_output = self.dropout(outputs[0])
        sequence_output = outputs[0]

        #embeds = self.embeddings(claimAuthorIX).view((1,-1))
        embeds = self.embeddings(claimAuthorIX)
        lin = F.relu(self.linearEmbs(embeds))
        out = sequence_output + lin 
        logits = F.log_softmax(out, dim=1)

        loss = None
        if labels is not None:

            #loss_fct = nn.CrossEntropyLoss()
            #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))
        
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions)