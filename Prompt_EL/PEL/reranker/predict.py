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
import torch.nn.functional as F

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PRE_TRAINED_MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
#PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
model = EntityRankerClassifier(2, PRE_TRAINED_MODEL_NAME)
model = model.to(device)

state_dict = torch.load('checkpoints/best_model_state.bin')
model.load_state_dict(state_dict)
print('Loading model parameters')

BATCH_SIZE = 20
MAX_LEN =128

tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

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

loss_fn = nn.CrossEntropyLoss().to(device)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    probs = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["label"].to(device)
            outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
          )
            prob = F.softmax(outputs,dim =1)
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)
            # breakpoint()r
            probs.extend(prob.cpu().numpy()[:,1].tolist())

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses), probs

val_acc, val_loss, probs= eval_model(
        model,
        val_data_loader,
        loss_fn, 
        device, 
        len(test_data)
      )
print(f'Val   loss {val_loss} accuracy {val_acc}')
print(probs)
with open('data/predict_scores.txt','w') as f:
    for texts, score in zip(list_data, probs):
      line = [texts['mention'],texts['entity'],str(texts['label']), str(texts['rank_score']),str(score)]
      f.write('\t'.join(line)+'\n')