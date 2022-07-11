import torch
from datasets import load_dataset
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
from getTextFromUrl import getCleanedTextFromUrl
from config import params
from tqdm import tqdm

class FormatDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, claimAuthors):
        self.encodings = encodings
        self.labels = labels
        self.claimAuthors = claimAuthors

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['claimAuthors'] = torch.tensor(self.claimAuthors[idx])
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def get_lists_from_dataset(dataset, author_to_ix):

    texts = []
    labels = []
    claimAuthors = []
    count=0
    claimAuthorsNames=[]

    for i in tqdm(dataset):
        if params['transformers_input_type'] == 'claim_text':
            texts.append(i[0])
        else:
            raise ValueError("Unknown option for input type : ", params['transformers_input_type'])

        labels.append(i[-1])
        if i[2] not in author_to_ix:
            claimAuthors.append(author_to_ix['unknown'])
        else:
            claimAuthors.append(author_to_ix[i[2]])
        claimAuthorsNames.append(i[2])
        count+=1
    return texts, labels, claimAuthors, claimAuthorsNames

def get_split_stat(dataset):
    classes_map = {}
    for row in dataset:
        label = row[-1]
        if label in classes_map:
            classes_map[label]+=1
        else:
            classes_map[label]=1
    return classes_map

def get_training_Authors(tData):

    _keys = set()
    for row in tData:
        _keys.add(row[2])
    _keys.add('unknown')
    
    author_to_ix = {author:i for i,author in enumerate(_keys)}
    return _keys, author_to_ix


def get_datasets():
    dataset = load_dataset("datacommons_factcheck", "fctchk_politifact_wapo")

    cleaned_dataset = []

    for row in dataset['train']:
        if row['review_rating'] in ['Pants on Fire', 'False', 'True']:
            if row['review_rating'] in ['Pants on Fire', 'False']:
                label=0
            else:
                label=1
            cleaned_dataset.append([row['claim_text'], row['review_url'], row['claim_author_name'], label])

    train_dataset = cleaned_dataset[:params['number_items_training_set']]
    eval_dataset = cleaned_dataset[params['number_items_training_set']:]

    train_classes_count = get_split_stat(train_dataset)
    print("stats train ", get_split_stat(train_dataset))
    print("stats val ", get_split_stat(eval_dataset))
    
    embedding_keys, author_to_ix = get_training_Authors(train_dataset)

    train_texts, train_labels, trainClaimAuthors,_ = get_lists_from_dataset(train_dataset, author_to_ix)
    val_texts, val_labels, valClaimAuthors, valClaimAuthorsNames = get_lists_from_dataset(eval_dataset, author_to_ix)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    train_dataset = FormatDataset(train_encodings, train_labels, trainClaimAuthors)
    val_dataset = FormatDataset(val_encodings, val_labels, valClaimAuthors)
    #return train_dataset,val_dataset,train_texts,val_texts,train_labels, val_labels, embedding_keys, author_to_ix, trainClaimAuthors, testClaimAuthors

    data_to_return = {}
    data_to_return['train_dataset'] = train_dataset
    data_to_return['val_dataset'] = val_dataset
    data_to_return['train_texts'] = train_texts
    data_to_return['val_texts'] = val_texts
    data_to_return['train_labels'] = train_labels
    data_to_return['val_labels'] = val_labels
    data_to_return['embedding_keys'] = embedding_keys
    data_to_return['author_to_ix'] = author_to_ix
    data_to_return['trainClaimAuthors'] = trainClaimAuthors
    data_to_return['valClaimAuthors'] = valClaimAuthors
    data_to_return['train_classes_count'] = train_classes_count
    data_to_return['valClaimAuthorsNames'] = valClaimAuthorsNames
    return data_to_return    

def getTokenizer():
    return tokenizer