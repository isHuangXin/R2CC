import os
import json
import random


READ_BASE_DIR = "/home/wanyao/huangxin/graphcodebert-cpp/dataset/cpp"
WRITE_BASE_DIR = "/home/wanyao/huangxin/graphcodebert-cpp/dataset/cpp-with-nageative"
random.seed(1234)

for i in range(3):
    all_dataset = []
    file_name = "data_{:04d}.parquet.json".format(i)
    negative_file_name = "data_{:04d}.negative.parquet.json".format(i)
    with open(os.path.join(READ_BASE_DIR, file_name), 'r', encoding='utf-8') as f_read:
        lines = f_read.readlines()
        for idx, line in enumerate(lines):
            code_nl_pair = json.loads(line)
            all_dataset.append(code_nl_pair)
    f_read.close()

    code_nl_pair_index = list(range(len(all_dataset)))
    is_used_flag = [1] * len(all_dataset)

    code_nl_pair_is_construct_dict = dict(zip(code_nl_pair_index, is_used_flag))
    code_negative_is_used_dict = dict(zip(code_nl_pair_index, is_used_flag))

    count = 0
    for i in range(len(all_dataset)):
        while code_nl_pair_is_construct_dict[i] == 1:
            random_choose_index = random.randint(0, len(all_dataset)-1)
            if all_dataset[random_choose_index]['loc'] != all_dataset[i]['loc'] and code_negative_is_used_dict[random_choose_index] == 1:
                all_dataset[i]['negative_function'] = all_dataset[random_choose_index]['function']
                all_dataset[i]['negative_loc'] = all_dataset[random_choose_index]['loc']
                code_nl_pair_is_construct_dict[i] = 0
                code_negative_is_used_dict[random_choose_index] = 0
            print(count)
            count += 1

    with open(os.path.join(WRITE_BASE_DIR, negative_file_name), 'w', encoding='utf-8') as f_write:
        for i in range(len(all_dataset)):
            json_str = json.dumps(all_dataset[i])
            f_write.write(json_str + "\n")
    f_write.close()

    print("negative construct finish" + negative_file_name)
