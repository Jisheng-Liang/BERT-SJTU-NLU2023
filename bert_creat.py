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
from transformers import (AutoTokenizer, AutoConfig, BertModel, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup, )
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from datasets import load_dataset, load_metric
from sklearn.model_selection import KFold
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')     # 这里会存储迄今最优模型的参数
        torch.save(model, 'finish_model.pkl')                 # 这里会存储迄今最优的模型
        self.val_loss_min = val_loss

class CustomDataset(Dataset):

    def __init__(self, data, maxlen, with_labels=True, bert_model='bert-base-uncased'):

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
    
def evaluate_loss(model, device, dataloader):
    model.eval()

    mean_loss = 0.0
    count = 0

    with torch.no_grad():
        for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(dataloader)):
            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
            outputs = model(input_ids=seq,
                            attention_mask=attn_masks,
                            token_type_ids=token_type_ids,
                            labels=labels,
                            output_hidden_states=True)
            loss = outputs[0].mean()
            mean_loss += loss.item()
            count += 1
    return mean_loss / count

class logit(BertModel):
    def __init__(self, config=AutoConfig.from_pretrained('bert-base-uncased', num_labels=3), add_pooling_layer: bool = True):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Initialize weights and apply final processing
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_embeddings(self,input_ids,token_type_ids):
        return self.embeddings(input_ids=input_ids,token_type_ids=token_type_ids)
    @autocast()
    def get_logits_from_embedding_output(self, embedding_output, attention_mask=None,
                                        labels=None, device=None):
        assert attention_mask.dim() == 2

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        outputs =  (sequence_output, pooled_output) + encoder_outputs[1:]
        
        # forSequenceClassification
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)
    @autocast()
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        
        outputs = (sequence_output, pooled_output) + encoder_outputs[1:]
        
        # forSequenceClassification
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class bert:
    def __init__(self, model, criterion, optimizer, scheduler, epochs, 
                 gradient_accumulation_steps,fp16=True,
                 ):

        self.model = model
        self.fp16 = fp16
        if self.fp16:
            self.scaler = GradScaler()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.criterion = criterion
    def step(self, dataset, bs, device):
        best_loss = np.Inf
        best_ep = 1
        nb_iterations = int(len(dataset)*0.8)
        print_every = nb_iterations // 5
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        for ep in range(self.epochs):

            # kFold cross-validation
            for fold, (train_index, val_index) in enumerate(kf.split(dataset)):
                train_fold = torch.utils.data.dataset.Subset(dataset, train_index)
                val_fold = torch.utils.data.dataset.Subset(dataset, val_index)

                train_loader = DataLoader(train_fold, batch_size=bs, num_workers=5)
                val_loader = DataLoader(val_fold, batch_size=bs, num_workers=5)
                self.model.train()
                train_loss = 0.0
                for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(train_loader)):
                    # Converting to cuda tensors
                    seq, attn_masks, token_type_ids, labels = \
                        seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)

                    # Enables autocasting for the forward pass (model + loss)
                    # Obtaining the logits from the model
                    with autocast():
                        outputs = self.model(seq, attn_masks, token_type_ids)
                
                        loss = self.criterion(outputs['logits'], labels)
                        # # Computing loss
                        loss = loss / self.gradient_accumulation_steps

                    if self.fp16:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                        
                    train_loss += loss.item()
                    if (it + 1) % self.gradient_accumulation_steps == 0:
                        if self.fp16:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                    
                    if (it + 1) % print_every == 0:  # Print training loss information
                        print()
                        print("No.{}Fold/{} of epoch {} complete. Loss : {} "
                            .format(fold, nb_iterations, ep+1, train_loss / print_every))
                        train_loss = 0.0


                val_loss = evaluate_loss(self.model, device, self.criterion, val_loader)  # Compute validation loss
                print()
                print("Epoch {} Fold {} complete! Validation Loss : {}".format(ep+1, fold, val_loss))

                if val_loss < best_loss:
                    print("Best validation loss improved from {} to {}".format(best_loss, val_loss))
                    print()
                    net_copy = copy.deepcopy(self.model)  # save a copy of the model
                    best_loss = val_loss
                    best_ep = ep + 1

        # Saving the model
        path_to_model='models/{}_lr_{}_val_loss_{}_ep_{}.pt'.format('bert', lr, round(best_loss, 5), best_ep)
        torch.save(net_copy.state_dict(), path_to_model)
        print("The model has been saved in {}".format(path_to_model))

        del loss
        torch.cuda.empty_cache()

def js_div(p, q):
    m = (p + q) / 2
    a = F.kl_div(p.log(), m, reduction='batchmean')
    b = F.kl_div(q.log(), m, reduction='batchmean')
    jsd = ((a + b) / 2)
    return jsd

def cos_loss(p, q):
    loss_fct = torch.nn.CosineSimilarity(dim=-1)
    return torch.sum(loss_fct(p, q), -1).mean()
class bert_creat:
    def __init__(self, model, optimizer, scheduler, epochs, 
                gradient_accumulation_steps, early_stopping,
                fp16=True,
                adv_steps=2,
                adv_lr=1e-1,
                adv_max_norm=1e-1,
                adv_temp=1.0,
                adv_init_var=1e-2
                ):

        self.model = model
        self.model_uw = model.module if hasattr(model, "module") else model
        self.word_embeddings = getattr(self.model_uw, self.model_uw.config.model_type.split("-")[0]).embeddings.word_embeddings
        self.early_stopping = early_stopping
        self.n_gpu = 1
        self.fp16 = fp16
        if self.fp16:
            self.scaler = GradScaler()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.adv_steps=adv_steps
        self.adv_lr=adv_lr
        self.adv_max_norm=adv_max_norm
        self.adv_temp=adv_temp
        self.adv_init_var=adv_init_var

    def _inner_update(self, delta, loss):
        delta_grad, = torch.autograd.grad(loss, delta)
        _shape = None
        if delta.dim() > 3:
            # e.g. multi-choice
            _shape = delta.shape
            delta, delta_grad = delta.view(-1, _shape[-2], _shape[-1]), delta_grad.view(-1, _shape[-2], _shape[-1])

        grad_norm = torch.norm(delta_grad.view(delta_grad.shape[0], -1), dim=-1, p="fro")
        grad_norm = torch.clamp(grad_norm, min=1e-8).view(-1, 1, 1)
        delta = (delta + self.adv_lr * delta_grad / grad_norm).detach()

        delta_norm = torch.norm(delta.view(delta.shape[0], -1), dim=-1, p="fro").detach()
        clip_mask = (delta_norm > self.adv_max_norm).to(delta)
        clip_weights = self.adv_max_norm / delta_norm * clip_mask + (1 - clip_mask)
        delta = (delta * clip_weights.view(-1, 1, 1)).detach()

        if _shape is not None:
            delta = delta.view(_shape)

        return delta

    def step(self, dataset, bs, device):
        best_loss = np.Inf
        best_ep = 1
        nb_iterations = int(len(dataset)*0.8)
        print_every = nb_iterations // 5
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        for ep in range(self.epochs):
            # kFold cross-validation
            for fold, (train_index, val_index) in enumerate(kf.split(dataset)):
                train_fold = torch.utils.data.dataset.Subset(dataset, train_index)
                val_fold = torch.utils.data.dataset.Subset(dataset, val_index)

                train_loader = DataLoader(train_fold, batch_size=bs, num_workers=5)
                val_loader = DataLoader(val_fold, batch_size=bs, num_workers=5)
                self.model.train()
                train_loss = 0.0
                for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(train_loader)):
                    # Converting to cuda tensors
                    seq, attn_masks, token_type_ids, labels = \
                        seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
                
                    inputs_embeds = self.word_embeddings(seq)
                    extended_input_mask = attn_masks.view(-1, attn_masks.size(-1)).unsqueeze(-1)
                    with autocast():
                        outputs = self.model(inputs_embeds=inputs_embeds,
                                                attention_mask=attn_masks,
                                                token_type_ids=token_type_ids,
                                                labels=labels,
                                                output_hidden_states=True)
                    loss = outputs[0].mean()
                    ctxr = outputs[-1][-1][:, 0]

                    delta = torch.zeros_like(inputs_embeds).normal_(0, 1) * self.adv_init_var
                    delta.requires_grad_()
                    for j in range(self.adv_steps):
                        with autocast():
                            outputs = self.model(inputs_embeds=inputs_embeds + delta,
                                                attention_mask=attn_masks,
                                                token_type_ids=token_type_ids,
                                                labels=labels,
                                                output_hidden_states=True)
                        loss_ptb = outputs[0].mean()
                        ctxr_ptb = outputs[-1][-1][:, 0]

                        if j == self.adv_steps - 1:
                            break

                        loss_ptb = loss_ptb - cos_loss(ctxr_ptb, ctxr.detach()) * self.adv_temp
                        delta = self._inner_update(delta, loss_ptb)
                        delta.requires_grad_()

                    loss = 0.5 * (loss + loss_ptb)

                    self.scaler.scale(loss).backward()
                    
                    train_loss += loss.item()

                    if (it + 1) % self.gradient_accumulation_steps == 0 or it == len(train_loader) - 1:
                        self.scaler.unscale_(self.optimizer)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.model.zero_grad()

                    if (it + 1) % print_every == 0:  # Print training loss information
                        print()
                        print("Iteration {}/{} of epoch {} complete. Loss : {} "
                            .format(it+1, len(train_loader), ep+1, train_loss / 5))
                        train_loss = 0.0


                val_loss = evaluate_loss(self.model, device, val_loader)  # Compute validation loss
                print()
                print("Epoch {} complete! Validation Loss : {}".format(ep+1, val_loss))

                if val_loss < best_loss:
                    print("Best validation loss improved from {} to {}".format(best_loss, val_loss))
                    print()
                    net_copy = copy.deepcopy(self.model)  # save a copy of the model
                    best_loss = val_loss
                    best_ep = ep + 1
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early Stopping!")
                break
        # Saving the model
        path_to_model='models/{}_lr_{}_val_loss_{}_ep_{}.pt'.format('bert', '2e-5', round(best_loss, 5), best_ep)
        torch.save(net_copy.state_dict(), path_to_model)
        print("The model has been saved in {}".format(path_to_model))

        del loss
        torch.cuda.empty_cache()

def main():
    # parameters
    maxlen = 128
    bert_model = "bert-base-uncased"  # 'albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2', 'albert-xxlarge-v2', 'bert-base-uncased', ...
    bs = 128  # batch size
    iters_to_accumulate = 2  # the gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate. If set to "1", you get the usual batch size
    lr = 2e-5  # learning rate
    epochs = 8  # number of training epochs
    num_labels = 3
    set_seed(1)

    # dataset 
    df = pd.read_csv('data/train.tsv', delimiter='\t')
    dataset = CustomDataset(df, maxlen, bert_model)

    # df_train = df.sample(frac=0.9, axis=0)
    # df_val = df[~df.index.isin(df_train.index)]
    # # Creating instances of training and validation set
    # print("Reading training data...")
    # train_set = CustomDataset(df_train, maxlen, bert_model)
    # print("Reading validation data...")
    # val_set = CustomDataset(df_val, maxlen, bert_model)
    # # Creating instances of training and validation dataloaders
    # train_loader = DataLoader(train_set, batch_size=bs, num_workers=5)
    # val_loader = DataLoader(val_set, batch_size=bs, num_workers=5)

    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(bert_model, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(bert_model,config=config)
    opti = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    early_stopping = EarlyStopping(patience=1, verbose=True)
    model.to(device)

    if torch.cuda.device_count() > 1:  # if multiple GPUs
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    num_warmup_steps = 0 # The number of steps for the warmup phase.
    t_total = int((len(dataset) * 0.8) // iters_to_accumulate) * epochs  # Necessary to take into account Gradient accumulation
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)
    
    trainer = bert_creat(model, opti, lr_scheduler, epochs, iters_to_accumulate, early_stopping)
    trainer.step(dataset, bs, device)
if __name__ == "__main__":
    main()