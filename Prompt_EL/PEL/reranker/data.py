
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import json
from tqdm import tqdm 

def read_data(data_path):
  list_data=[]
  #with open(data_path, 'r', encoding='utf-8') as f:
  with open(data_path) as f:
    for item in f.readlines():
      arr = item.strip().split('\t')
      list_data.append({
        'mention':arr[0],
        'entity':arr[1],
        'label':int(arr[3])
      })
  return list_data

def read_context_data(data_path):
  list_data=[]
  with open(data_path) as f:
    for item in f.readlines():
      arr = item.strip().split('\t')
      list_data.append({
        'mention':arr[0],
        'entity':arr[1],
        'entity_id':arr[2],
        'context':arr[3],
        'label':int(arr[-1])
      })
  return list_data


def check_k(queries):
  return len(queries[0]['mentions'][0]['candidates'])

def read_json_data(data_path):
  list_data=[]
  with open(data_path) as f:
    data= json.loads(f.read())

    queries = data['queries']

    for query in tqdm(queries):
      mentions = query['mentions']
      for mention in mentions:
        candidates = mention['candidates']
        mention_name = mention["mention"]
        for candidate in candidates:
          name=candidate["name"]
          label = candidate['label']
          rank_score = candidate['rank_score']
          context = candidate['context']
          # breakpoint()
          list_data.append(
            {
              'mention':mention_name,
              'entity':name,
              'rank_score':rank_score,
              'label':int(label),
              'context': context
            }
          )
  return list_data




class MentionEntityDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):

      self.tokenizer = tokenizer
      self.max_len = max_len
      self.data = data
  
    def __len__(self):
      return len(self.data)
  
    def __getitem__(self, index):
      # breakpoint()
      example = self.data[index]
      mention = example["mention"]
      entity = example["entity"]
      if 'context' in example:
        context = example['context']
      label = example["label"]
      prompt_texts = 'is the ' + mention + ' similar to the ' + entity+'?'
      #prompt_texts = mention + ' and '+entity+ 'are similar?'
      #print(prompt_texts)
      # print(context)
      encoding = self.tokenizer.encode_plus(
        prompt_texts,
        text_pair=context,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt',
      )
      # prompt_texts = mention + ' '+entity+' is similar'
      # encoding = self.tokenizer.encode_plus(
      #   mention,
      #   text_pair=entity,
      #   add_special_tokens=True,
      #   max_length=self.max_len,
      #   return_token_type_ids=False,
      #   padding='max_length',
      #   return_attention_mask=True,
      #   truncation=True,
      #   return_tensors='pt',
      # )
      tokenized_inputs = {
        "input_ids":encoding["input_ids"][0],
        "attention_mask":encoding["attention_mask"][0],
        "label":label
      }
      tokenized_inputs.update({'label':label})
      return tokenized_inputs


if __name__=="__main__":
  BATCH_SIZE = 16
  MAX_LEN = 128
  PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
  tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
  data_path = '../output/mention_entity_pairs.txt'
  list_data = read_data(data_path)
  train_data = MentionEntityDataset(list_data, tokenizer=tokenizer, max_len=16)
  # print(train_data[0])
  train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        num_workers=2
      )
  for batch in train_loader:
    print(batch)
    break

    