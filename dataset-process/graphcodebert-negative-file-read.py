import json
import os

WRITE_BASE_DIR = "/home/wanyao/huangxin/graphcodebert-cpp/dataset/java_codeserachnet_process_dataset_with_negative"

# train_with_negative.jsonl

all_dataset = []
with open(os.path.join(WRITE_BASE_DIR, "train_with_negative.jsonl"), 'r', encoding='utf-8') as f_read:
    lines = f_read.readlines()
    for idx, line in enumerate(lines):
        data = json.loads(line)
        all_dataset.append(data)
f_read.close()