import torch
import torch.nn as nn
import os
import random
import numpy as np
import pandas as pd
import copy
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import (AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup)
from datasets import load_dataset, load_metric
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CustomDataset(Dataset):

    def __init__(self, data, maxlen, with_labels=True, bert_model='microsoft/deberta-v3-large'):

        self.data = data  # pandas dataframe
        #Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)  

        self.maxlen = maxlen
        self.with_labels = with_labels 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent1 = str(self.data.iloc[index,1])
        sent2 = str(self.data.iloc[index,2])

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(sent1, sent2, 
                                      padding='max_length',  # Pad to max_length
                                      truncation='longest_first',  # Truncate to max_length
                                      max_length=self.maxlen,  
                                      return_tensors='pt')  # Return torch.Tensor objects
        
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels
            label = self.data.iloc[index,3]
            return token_ids, attn_masks, token_type_ids, label  
        else:
            return token_ids, attn_masks, token_type_ids

def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def evaluate_loss(model, device, criterion, dataloader):
    model.eval()

    mean_loss = 0.0
    count = 0

    with torch.no_grad():
        for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(dataloader)):
            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
            outputs = model(input_ids=seq,
                            attention_mask=attn_masks,
                            token_type_ids=token_type_ids,)
            mean_loss += criterion(outputs['logits'], labels).item()
            count += 1
    return mean_loss / count

# class SentencePairClassifier(nn.Module):

#     def __init__(self, bert_model="albert-base-v2", freeze_bert=False):
#         super(SentencePairClassifier, self).__init__()
#         #  Instantiating BERT-based model object

#         self.bert_layer = AutoModel.from_pretrained(bert_model)
#         #  Fix the hidden-state size of the encoder outputs (If you want to add other pre-trained models here, search for the encoder output size)
#         if bert_model == "albert-base-v2":  # 12M parameters
#             hidden_size = 768
#         elif bert_model == "albert-large-v2":  # 18M parameters
#             hidden_size = 1024
#         elif bert_model == "albert-xlarge-v2":  # 60M parameters
#             hidden_size = 2048
#         elif bert_model == "albert-xxlarge-v2":  # 235M parameters
#             hidden_size = 4096
#         elif bert_model == "bert-base-uncased": # 110M parameters
#             hidden_size = 768
#         elif bert_model == "microsoft/deberta-v3-large": # 110M parameters
#             hidden_size = 1024,
#             self.bert_layer = DebertaV2Model.from_pretrained(bert_model)

#         # Freeze bert layers and only train the classification layer weights
#         if freeze_bert:
#             for p in self.bert_layer.parameters():
#                 p.requires_grad = False

#         # Classification layer
#         self.cls_layer = nn.Linear(hidden_size, 3)
#         self.act = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.1)

#     @autocast()  # run in mixed precision
#     def forward(self, input_ids, attn_masks, token_type_ids):
#         '''
#         Inputs:
#             -input_ids : Tensor  containing token ids
#             -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
#             -token_type_ids : Tensor containing token type ids to be used to identify sentence1 and sentence2
#         '''

#         # Feeding the inputs to the BERT-based model to obtain contextualized representations
#         outputs = self.bert_layer(input_ids, attn_masks, token_type_ids, return_dict=False)
#         pooler_output = self.act(outputs[1])
#         # Feeding to the classifier layer the last layer hidden-state of the [CLS] token further processed by a
#         # Linear Layer and a Tanh activation. The Linear layer weights were trained from the sentence order prediction (ALBERT) or next sentence prediction (BERT)
#         # objective during pre-training.
#         logits = self.cls_layer(self.dropout(pooler_output))

#         return logits

def get_probs_from_logits(logits):
    """
    Converts a tensor of logits into an array of probabilities by applying the sigmoid function
    """
    probs = torch.sigmoid(logits.unsqueeze(-1))
    return probs.detach().cpu().numpy()

def test_prediction(net, device, dataloader, with_labels=True, result_file="results/output.txt"):
    """
    Predict the probabilities on a dataset with or without labels and print the result in a file
    """
    net.eval()
    w = open(result_file, 'w')
    probs_all = []

    with torch.no_grad():
        if with_labels:
            for seq, attn_masks, token_type_ids, _ in tqdm(dataloader):
                seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
                outputs = net(seq, attn_masks, token_type_ids)
                probs = get_probs_from_logits(outputs['logits']).squeeze(-1)
                probs_all += probs.tolist()
        else:
            for seq, attn_masks, token_type_ids in tqdm(dataloader):
                seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
                outputs = net(seq, attn_masks, token_type_ids)
                probs = get_probs_from_logits(outputs['logits']).squeeze(-1)
                probs_all += probs.tolist()

    w.writelines(str(prob)+'\n' for prob in probs_all)
    w.close()

def edit(output_path):
    df = pd.read_csv('data/sample_submission.csv')
    f = open(output_path,'r')
    i = 0
    while True:
        line = f.readline()
        if line:
            line = line.strip('[')
            line = line.strip(']\n')
            line = line.split(', ')
            label = line.index(max(np.array(line)))
            text = None
            if label == 1:
                text = 'contradiction'
            elif label == 0:
                text = 'entailment'
            elif label == 2:
                text = 'neutral'
            else:
                print(False)
            df.iloc[i,1] = text
            i += 1
        else:
            break
    f.close()
    df.to_csv(output_path,index=False)

def main():
    # parameters
    maxlen = 128
    bert_model = "bert-base-uncased"  # 'albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2', 'albert-xxlarge-v2', 'bert-base-uncased', ...
    bs = 64  # batch size
    num_labels = 3
    set_seed(1)

    # dataset 
    print("Reading test data...")
    df_test = pd.read_csv('data/test.tsv', delimiter='\t')
    test_set = CustomDataset(df_test, maxlen, False, bert_model)
    # Creating instances of training and validation dataloaders
    # train_loader = DataLoader(train_set, batch_size=bs, num_workers=5)
    # val_loader = DataLoader(val_set, batch_size=bs, num_workers=5)
    test_loader = DataLoader(test_set, batch_size=bs, num_workers=5)
    path_to_model = 'models/bert_lr_2e-5_val_loss_0.00108_ep_5.pt'
    path_to_output_file = 'results/output_creat.csv'


    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(bert_model, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(bert_model,config=config)
    model.to(device)
    if torch.cuda.device_count() > 1:  # if multiple GPUs
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    print()
    print("Loading the weights of the model...")
    model.load_state_dict(torch.load(path_to_model))
    model.to(device)
    

    test_prediction(net=model, device=device, dataloader=test_loader, with_labels=False,  # set the with_labels parameter to False if your want to get predictions on a dataset without labels
                result_file=path_to_output_file)
    edit(path_to_output_file)
    print()
    print("Predictions are available in : {}".format(path_to_output_file))
if __name__ == "__main__":
    main()