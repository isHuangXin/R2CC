import os
import json


READ_BASE_DIR = "/home/wanyao/huangxin/graphcodebert-cpp/dataset/cpp-with-nageative"
WRITE_BASE_DIR = "/home/wanyao/huangxin/graphcodebert-cpp/dataset/cpp-dataset-split"

url_index = 0
all_dataset = []
train_dataset = []
valid_dataset = []
test_dataset = []
codebase_dataset = []
for i in range(3):
    file_name = "data_{:04d}.negative.parquet.json".format(i)
    with open(os.path.join(READ_BASE_DIR, file_name), 'r', encoding='utf-8') as f_read:
        lines = f_read.readlines()
        for idx, line in enumerate(lines):
            code_nl_pair = json.loads(line)
            code_nl_pair['url'] = url_index
            url_index += 1
            all_dataset.append(code_nl_pair)
    f_read.close()
print("all dataset load done!")

for i in range(len(all_dataset)):
    if i < 207887:
        train_dataset.append(all_dataset[i])
    else:
        codebase_dataset.append(all_dataset[i])
        if 207887 <= i < 213070:
            valid_dataset.append(all_dataset[i])
        if 213070 <= i < 224025:
            test_dataset.append(all_dataset[i])
print("train, valid, test, codebase split done!")

# train dataset write to json
with open(os.path.join(WRITE_BASE_DIR, "train_with_negative.jsonl"), 'w', encoding='utf-8') as f_write:
    for i in range(len(train_dataset)):
        json_str = json.dumps(train_dataset[i])
        f_write.write(json_str + "\n")
f_write.close()
print("train dataset write to the json file")

# train dataset write to json
with open(os.path.join(WRITE_BASE_DIR, "train_with_negative.jsonl"), 'w', encoding='utf-8') as f_write:
    for i in range(len(train_dataset)):
        json_str = json.dumps(train_dataset[i])
        f_write.write(json_str + "\n")
f_write.close()
print("train dataset write to the json file")

# valid dataset write to json
with open(os.path.join(WRITE_BASE_DIR, "valid_with_negative.jsonl"), 'w', encoding='utf-8') as f_write:
    for i in range(len(valid_dataset)):
        json_str = json.dumps(valid_dataset[i])
        f_write.write(json_str + "\n")
f_write.close()
print("valid dataset write to the json file")

# test dataset write to json
with open(os.path.join(WRITE_BASE_DIR, "test_with_negative.jsonl"), 'w', encoding='utf-8') as f_write:
    for i in range(len(test_dataset)):
        json_str = json.dumps(test_dataset[i])
        f_write.write(json_str + "\n")
f_write.close()
print("test dataset write to the json file")

# codebase dataset write to json
with open(os.path.join(WRITE_BASE_DIR, "codebase_with_negative.jsonl"), 'w', encoding='utf-8') as f_write:
    for i in range(len(codebase_dataset)):
        json_str = json.dumps(codebase_dataset[i])
        f_write.write(json_str + "\n")
f_write.close()
print("codebase dataset write to the json file")