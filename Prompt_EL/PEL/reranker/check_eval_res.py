
import csv
import json
import numpy as np
import pdb
from tqdm import tqdm
import logging

def check_k(queries):
    return len(queries[0]['mentions'][0]['candidates'])

def evaluate_topk_acc(data):
    """
    evaluate acc@1~acc@k
    """
    queries = data['queries']
    k = check_k(queries)

    for i in range(0, k):
        hit = 0
        for query in queries:
            mentions = query['mentions']
            mention_hit = 0
            for mention in mentions:
                candidates = mention['candidates'][:i+1] # to get acc@(i+1)
                mention_hit += np.any([candidate['label'] for candidate in candidates])
            
            # When all mentions in a query are predicted correctly,
            # we consider it as a hit 
            if mention_hit == len(mentions):
                hit +=1
        
        data['acc{}'.format(i+1)] = hit/len(queries)

    return data


LOGGER = logging.getLogger()
# predictions_eval
# predictions
with open('./data/predictions.json') as f:
    result= json.loads(f.read())

res = evaluate_topk_acc(result)
print("acc@1={}".format(res['acc1']))
print("acc@5={}".format(res['acc5']))
# print(res)