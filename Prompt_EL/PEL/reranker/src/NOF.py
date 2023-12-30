import json

# 读取JSON文件
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# 递归提取键为"label"的数据并统计0和1的数量
def count_label_data(data):
    count_0 = 0
    count_1 = 0

    if isinstance(data, dict):
        if 'label' in data:
            label = str(data['label'])
            count_0 += label.count('0')
            count_1 += label.count('1')
        for value in data.values():
            count_0_child, count_1_child = count_label_data(value)
            count_0 += count_0_child
            count_1 += count_1_child
    elif isinstance(data, list):
        for item in data:
            count_0_child, count_1_child = count_label_data(item)
            count_0 += count_0_child
            count_1 += count_1_child

    return count_0, count_1

# 示例JSON文件路径
file_path = 'F:\Biosyn1/reranker\data\predictions_train.json'

# 读取JSON文件
data = read_json_file(file_path)

# 提取键为"label"的数据并统计0和1的数量
count_0, count_1 = count_label_data(data)

# 打印结果
print("Count of '0':", count_0)
print("Count of '1':", count_1)


