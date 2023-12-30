import os

import transformers
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from model import EntityRankerClassifier
from data import read_data, MentionEntityDataset, read_context_data, read_json_data

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PRE_TRAINED_MODEL_NAME = './emilyalsentzer/Bio_ClinicalBERT'
#PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
#PRE_TRAINED_MODEL_NAME = 'dmis-lab/biobert-large-cased-v1.1'
checkpoint_path = 'checkpoints'
os.makedirs(checkpoint_path, exist_ok=True)

model = EntityRankerClassifier(2, PRE_TRAINED_MODEL_NAME)
model = model.to(device)
EPOCHS = 20

BATCH_SIZE =20
MAX_LEN = 128

tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
# data_path = '../output/train_mention_entity_pairs.txt'
# list_data = read_data(data_path)
# list_data = read_context_data(data_path)

data_path = './data/predictions_train.json'
list_data = read_json_data(data_path)
train_data = MentionEntityDataset(list_data, tokenizer=tokenizer, max_len=MAX_LEN)
train_data_loader = DataLoader(
      train_data,
      batch_size=BATCH_SIZE,
      num_workers=0
    )

# data_path = '../output/test_mention_entity_pairs.txt'
# list_data = read_data(data_path)
# list_data = read_context_data(data_path)
data_path ='./data/predictions_eval.json'
list_data = read_json_data(data_path)
test_data = MentionEntityDataset(list_data, tokenizer=tokenizer, max_len=MAX_LEN)
val_data_loader = DataLoader(
      test_data,
      batch_size=BATCH_SIZE,
      num_workers=0
    )

optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)
# label smoothing
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler, 
  n_examples
):
    model = model.train()

    losses = []
    correct_predictions = 0
  
    for d in data_loader:

        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["label"].to(device)

        outputs = model(
                      input_ids=input_ids,
                      attention_mask=attention_mask
                        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)


        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        # backward
        #print(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    losses = []
    correct_predictions = 0
    model = model.eval()
    # 固定dropout，不产生梯度
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["label"].to(device)
            outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
          )
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


history = defaultdict(list)
best_accuracy = 0
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,    
        loss_fn, 
        optimizer, 
        device, 
        scheduler, 
        len(train_data)
      )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn, 
        device, 
        len(test_data)
      )

    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
      save_path = os.path.join(checkpoint_path,'best_model_state.bin')
      torch.save(model.state_dict(), save_path)
      print('save model {}'.format(save_path))
      best_accuracy = val_acc
