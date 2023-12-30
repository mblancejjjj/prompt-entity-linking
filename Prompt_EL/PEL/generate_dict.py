
import collections

d = collections.defaultdict(list)
src_path = '../NCBI/train_data.txt'
tgt_path = '../NCBI/train_dictionary.txt'

list_data = []
with open(src_path) as f:
    for item in f.readlines():
        arr = item.strip().split('\t')
        entity_id, entity = arr[2],arr[3]
        if entity not in d:
            list_data.append([entity_id,entity])
            d[entity]=entity_id

src_path = '../NCBI/test_data.txt'
with open(src_path) as f:
    for item in f.readlines():
        arr = item.strip().split('\t')
        entity_id, entity = arr[2],arr[3]
        if entity not in d:
            list_data.append([entity_id,entity])
            d[entity]=entity_id

with open(tgt_path,'w') as f:
    for item in list_data:
        f.write('\t'.join(item)+'\n')
