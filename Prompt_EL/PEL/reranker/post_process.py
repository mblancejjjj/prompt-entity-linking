import json
import os

top_k = 20
list_data = []
with open('./data/predict_scores.txt') as f:
    data = []
    for i, item in enumerate(f.readlines()):
        arr = item.strip().split('\t')
        rank_score = float(arr[-2])
        alpha =0.8
        weight_avg = float(arr[-1]) * alpha + rank_score * (1 - alpha)
        # weight_avg = float(arr[-1])*0.2+rank_score
        # weight_avg = rank_score
        # weight_avg = float(arr[-1])
        # print(weight_avg)
        arr.append(weight_avg)
        data.append(arr)
        if (len(data) == top_k):
            # print(data)
            # breakpoint()
            data = sorted(data, key=lambda x: float(x[-1]), reverse=True)
            list_data.append(data)
            data = []

print(list_data[:2])
# breakpoint()
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
# breakpoint()

output_dir = 'data'
output_file = os.path.join(output_dir, "predictions.json")
with open(output_file, 'w') as f:
    json.dump(result, f, indent=2)

# with open('./output/ranking_predict.txt','w') as f:
#     for data in list_data:
#         line=[]
#         for arr in data:
#             line += [arr[2],arr[1]]

#         f.write('\t'.join(line)+'\n')



