import json
import os
import random

random.seed(1234)

READ_BASE_DIR = "/home/wanyao/huangxin/graphcodebert-cpp/dataset/java_codeserachnet_process_dataset"
WRITE_BASE_DIR = "/home/wanyao/huangxin/graphcodebert-cpp/dataset/java_codeserachnet_process_dataset_with_negative"

# codebase.jsonl  test.jsonl  train.jsonl  valid.jsonl

all_dataset = []
with open(os.path.join(READ_BASE_DIR, "codebase.jsonl"), 'r', encoding='utf-8') as f_read:
    lines = f_read.readlines()
    for idx, line in enumerate(lines):
        data = json.loads(line)
        all_dataset.append(data)
f_read.close()

all_code_pair_index = list(range(len(all_dataset)))
nl_code_pair_is_constructed = [1] * len(all_dataset)
nl_code_pair_is_constructed_dict = dict(zip(all_code_pair_index, nl_code_pair_is_constructed))

negative_is_used = [1] * len(all_dataset)
negative_is_used_dict = dict(zip(all_code_pair_index, negative_is_used))

count = 0
for i in range(len(all_dataset)):
    while nl_code_pair_is_constructed_dict[i] == 1:
        random_choose_index = random.randint(0, len(all_dataset) - 1)
        if negative_is_used_dict[random_choose_index] == 1 and random_choose_index != i:
            all_dataset[i]['negative_original_string'] = all_dataset[random_choose_index]['original_string']
            negative_is_used_dict[random_choose_index] = 0
            nl_code_pair_is_constructed_dict[i] = 0
        print(count)
        count += 1
print("negative pair construct done!")

# dataset write the json file
with open(os.path.join(WRITE_BASE_DIR, "codebase_with_negative.jsonl"), 'w', encoding='utf-8') as f_write:
    for i in range(len(all_dataset)):
        json_str = json.dumps(all_dataset[i])
        f_write.write(json_str + "\n")
print("dataset {} pairs has written to json".format(len(all_dataset)))
f_write.close()