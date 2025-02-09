import json
import pandas as pd

# 读取原始 JSON 文件
with open('falcon-few-shot.json', 'r') as file:
    data = json.load(file)

# 创建一个新的列表来存储修改后的数据
new_data = []

# 遍历原始数据，提取并格式化信息
for key, value in data.items():
    # 删除所有的换行符
    value = value.replace('\n', '')

    # 提取 question, choices, answer, 和 reasoning
    question_start = value.find("Question:") + len("Question:")
    choices_start = value.find("Choices:")
    answer_start = value.find("Answer:")
    reasoning_start = value.find("Reasoning:")

    question = value[question_start:choices_start].strip()
    choices = value[choices_start + len("Choices:"):answer_start].strip()
    answer = value[answer_start + len("Answer:"):reasoning_start].strip()
    reasoning = value[reasoning_start + len("Reasoning:"):].strip()

    # 将提取的数据存储在新的字典中
    formatted_entry = {
        'question': question,
        'choices': choices,
        'answer': answer,
        'reasoning': reasoning
    }

    # 将格式化后的条目添加到新列表中
    new_data.append(formatted_entry)

# 读取 CSV 文件
csv_data = pd.read_csv('ECQA-IR.csv')

# 遍历 CSV 中的每一行，合并数据
for _, row in csv_data.iterrows():
    question = row['question']
    gold_label = row['gold_label']
    ground_truth = row['ground_truth']
    id_value = row['id']

    # 在 JSON 数据中查找匹配的 question
    for entry in new_data:
        if entry['question'] == question:
            # 添加新的字段到 JSON 数据
            entry['gold_label'] = gold_label
            entry['ground_truth'] = ground_truth
            entry['id'] = id_value

# 写回 JSON 文件
with open('modified.json', 'w') as json_file:
    json.dump(new_data, json_file, ensure_ascii=False, indent=4)

print("文件修改并保存成功。")
