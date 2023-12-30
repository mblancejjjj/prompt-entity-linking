import json
import os
import csv
import numpy as np
import logging
from tqdm import tqdm


def check_k(queries):
    return len(queries[0]['mentions'][0]['candidates'])


def evaluate_topk_acc(data):
    queries = data['queries']
    k = check_k(queries)

    for i in range(0, k):
        hit = 0
        for query in queries:
            mentions = query['mentions']
            mention_hit = 0
            for mention in mentions:
                candidates = mention['candidates'][:i + 1]  # to get acc@(i+1)
                mention_hit += np.any([candidate['label'] for candidate in candidates])

            # When all mentions in a query are predicted correctly,
            # we consider it as a hit
            if mention_hit == len(mentions):
                hit += 1

        data['acc{}'.format(i + 1)] = hit / len(queries)

    return data


LOGGER = logging.getLogger()

# Read the first file
top_k = 20
list_data = []
with open('./data/predict_scores.txt') as f:
    data = []
    for i, item in enumerate(f.readlines()):
        arr = item.strip().split('\t')
        rank_score = float(arr[-2])

        for alpha in np.arange(0, 1.01, 0.01):
            weight_avg = float(arr[-1]) * alpha + rank_score * (1 - alpha)
            arr.append(weight_avg)
            data.append(arr)
            if len(data) == top_k:
                data = sorted(data, key=lambda x: float(x[-1]), reverse=True)
                list_data.append(data)
                data = []

# Prepare data for the second file
queries = []
for data in list_data:
    dict_mentions = []
    dict_candidates = []
    for arr in data:
        dict_candidates.append({
            'name': arr[1],
            'label': int(arr[2])
        })
    dict_mentions.append({
        'mention': data[0][0],
        'candidates': dict_candidates
    })

    queries.append({
        'mentions': dict_mentions
    })

result = {
    'queries': queries
}

# Write the merged data to predictions.json
output_dir = 'data'
output_file = os.path.join(output_dir, "predictions.json")
with open(output_file, 'w') as f:
    json.dump(result, f, indent=2)


# Evaluate the merged data
def evaluate_topk_acc(data):
    queries = data['queries']
    k = check_k(queries)

    for i in range(0, k):
        hit = 0
        for query in queries:
            mentions = query['mentions']
            mention_hit = 0
            for mention in mentions:
                candidates = mention['candidates'][:i + 1]  # to get acc@(i+1)
                mention_hit += np.any([candidate['label'] for candidate in candidates])

            # When all mentions in a query are predicted correctly,
            # we consider it as a hit
            if mention_hit == len(mentions):
                hit += 1

        data['acc{}'.format(i + 1)] = hit / len(queries)

    return data


# Evaluate the merged data
with open('./data/predictions.json') as f:
    result = json.loads(f.read())

# Output the alpha values and evaluation results
output_file = os.path.join(output_dir, "alpha_evaluation.csv")
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Alpha', 'acc@1', 'acc@5'])
    for alpha in np.arange(0, 1.01, 0.01):
        res = evaluate_topk_acc(result)
        writer.writerow([alpha, res['acc1'], res['acc5']])
        print("Alpha: {:.2f} | acc@1: {:.4f} | acc@5: {:.4f}".format(alpha, res['acc1'], res['acc5']))

