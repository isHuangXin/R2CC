```shell
lang=ruby
mkdir -p ./saved_models/java
python run.py \
    --output_dir=./saved_models/java \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --lang=java \
    --do_train \
    --train_data_file=/mnt/silver/huangxin/dataset/codesearchnet/java-for-codebert-vector/train_with_negative.json \
    --eval_data_file=/mnt/silver/huangxin/dataset/codesearchnet/java-for-codebert-vector/valid_with_negative.json \
    --test_data_file=/mnt/silver/huangxin/dataset/codesearchnet/java-for-codebert-vector/test_with_negative.json \
    --codebase_file=/mnt/silver/huangxin/dataset/codesearchnet/java-for-codebert-vector/valid_with_negative.json \
    --num_train_epochs 10 \
    --code_length 256 \
    --data_flow_length 64 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456
```
```shell
(py37-1.7) ➜  codesearch nohup python run.py --output_dir=./saved_models/java --config_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base --tokenizer_name=microsoft/graphcodebert-base --lang=java --do_train --train_data_file=/mnt/silver/huangxin/dataset/codesearchnet/java-for-codebert-vector/train_with_negative.json --eval_data_file=/mnt/silver/huangxin/dataset/codesearchnet/java-for-codebert-vector/valid_with_negative.json --test_data_file=/mnt/silver/huangxin/dataset/codesearchnet/java-for-codebert-vector/test_with_negative.json --codebase_file=/mnt/silver/huangxin/dataset/codesearchnet/java-for-codebert-vector/valid_with_negative.json --num_train_epochs 10 --code_length 256 --data_flow_length 64 --nl_length 128 --train_batch_size 64 --eval_batch_size 64 --learning_rate 2e-5 --seed 123456 > /mnt/silver/huangxin/source_code/codebert/GraphCodeBERT/codesearch/txt/train_1.log 2>&1 &
[1] 51867 
```


### official
```shell
(py37-1.7) ➜  codesearch nohup python run.py --output_dir=./saved_models/java-official-feature --config_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base --tokenizer_name=microsoft/graphcodebert-base --lang=java --do_train --train_data_file=/mnt/silver/huangxin/dataset/codesearchnet/java-for-codebert-vector/train_with_negative.json --eval_data_file=/mnt/silver/huangxin/dataset/codesearchnet/java-for-codebert-vector/valid_with_negative.json --test_data_file=/mnt/silver/huangxin/dataset/codesearchnet/java-for-codebert-vector/test_with_negative.json --codebase_file=/mnt/silver/huangxin/dataset/codesearchnet/java-for-codebert-vector/valid_with_negative.json --num_train_epochs 10 --code_length 256 --data_flow_length 64 --nl_length 128 --train_batch_size 64 --eval_batch_size 64 --learning_rate 2e-5 --seed 123456 > /mnt/silver/huangxin/source_code/codebert/GraphCodeBERT/codesearch/txt/train_3.log 2>&1 &
[1] 36114
```

# codesearnet myself process

## train_1.log  code_tokens

## train_2.log origin_string 1000

## train_3.log origin_string all

# codesearchnet  graphcodebert official process

## train_4.log official dataset 1000

## train_5.log official dataset all

```shell
(py37-1.7) ➜  graphcodebert-cpp nohup python run.py --output_dir=./saved_models/java-official-dataset --config_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base --tokenizer_name=microsoft/graphcodebert-base --lang=java --do_train --train_data_file=/home/wanyao/huangxin/graphcodebert-cpp/dataset/java_codeserachnet_process_dataset/train.jsonl --eval_data_file=/home/wanyao/huangxin/graphcodebert-cpp/dataset/java_codeserachnet_process_dataset/valid.jsonl --test_data_file=/home/wanyao/huangxin/graphcodebert-cpp/dataset/java_codeserachnet_process_dataset/test.jsonl --codebase_file=/home/wanyao/huangxin/graphcodebert-cpp/dataset/java_codeserachnet_process_dataset/codebase.jsonl --num_train_epochs 10 --code_length 256 --data_flow_length 64 --nl_length 128 --train_batch_size 64 --eval_batch_size 64 --learning_rate 2e-5 --seed 123456 > /home/wanyao/huangxin/graphcodebert-cpp/txt/train_5.log 2>&1 &
[1] 46333
```

java-official-dataset-without-dfg

nohup python run.py --output_dir=./saved_models/java-official-dataset-without-dfg --config_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base --tokenizer_name=microsoft/graphcodebert-base --lang=java --do_train --train_data_file=/home/wanyao/huangxin/graphcodebert-cpp/dataset/java_codeserachnet_process_dataset/train.jsonl --eval_data_file=/home/wanyao/huangxin/graphcodebert-cpp/dataset/java_codeserachnet_process_dataset/valid.jsonl --test_data_file=/home/wanyao/huangxin/graphcodebert-cpp/dataset/java_codeserachnet_process_dataset/test.jsonl --codebase_file=/home/wanyao/huangxin/graphcodebert-cpp/dataset/java_codeserachnet_process_dataset/codebase.jsonl --num_train_epochs 10 --code_length 320 --data_flow_length 64 --nl_length 128 --train_batch_size 64 --eval_batch_size 64 --learning_rate 2e-5 --seed 123456 > /home/wanyao/huangxin/graphcodebert-cpp/txt/train_6.log 2>&1 &


```shell
train_6.log 
# 去掉了CFG的 graphcodebert
(py37-1.7) ➜  graphcodebert-cpp nohup python run.py --output_dir=./saved_models/java-official-dataset-without-dfg --config_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base --tokenizer_name=microsoft/graphcodebert-base --lang=java --do_train --train_data_file=/home/wanyao/huangxin/graphcodebert-cpp/dataset/java_codeserachnet_process_dataset/train.jsonl --eval_data_file=/home/wanyao/huangxin/graphcodebert-cpp/dataset/java_codeserachnet_process_dataset/valid.jsonl --test_data_file=/home/wanyao/huangxin/graphcodebert-cpp/dataset/java_codeserachnet_process_dataset/test.jsonl --codebase_file=/home/wanyao/huangxin/graphcodebert-cpp/dataset/java_codeserachnet_process_dataset/codebase.jsonl --num_train_epochs 10 --code_length 320 --data_flow_length 64 --nl_length 128 --train_batch_size 64 --eval_batch_size 64 --learning_rate 2e-5 --seed 123456 > /home/wanyao/huangxin/graphcodebert-cpp/txt/train_6.log 2>&1 &
[1] 35787
```

## for train_7.log trippet loss
```shell
(py37-1.7) ➜  graphcodebert-cpp nohup python run.py --output_dir=./saved_models/java-official-dataset-without-dfg-triple-loss --config_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base --tokenizer_name=microsoft/graphcodebert-base --lang=java --do_train --train_data_file=/home/wanyao/huangxin/graphcodebert-cpp/dataset/java_codeserachnet_process_dataset_with_negative/train_with_negative.jsonl --eval_data_file=/home/wanyao/huangxin/graphcodebert-cpp/dataset/java_codeserachnet_process_dataset_with_negative/valid_with_negative.jsonl --test_data_file=/home/wanyao/huangxin/graphcodebert-cpp/dataset/java_codeserachnet_process_dataset_with_negative/test_with_negative.jsonl --codebase_file=/home/wanyao/huangxin/graphcodebert-cpp/dataset/java_codeserachnet_process_dataset_with_negative/codebase_with_negative.jsonl --num_train_epochs 10 --code_length 320 --data_flow_length 64 --nl_length 128 --train_batch_size 64 --eval_batch_size 64 --learning_rate 2e-5 --seed 123456 > /home/wanyao/huangxin/graphcodebert-cpp/txt/train_7.log 2>&1 &
[1] 34727
```

## for train_8.log to check np.dot and np cosing similarity
```shell
(py37-1.7) ➜  graphcodebert-cpp nohup python run.py --output_dir=./saved_models/java-official-dataset-without-dfg-triple-loss --config_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base --tokenizer_name=microsoft/graphcodebert-base --lang=java --do_train --train_data_file=/home/wanyao/huangxin/graphcodebert-cpp/dataset/java_codeserachnet_process_dataset_with_negative/train_with_negative.jsonl --eval_data_file=/home/wanyao/huangxin/graphcodebert-cpp/dataset/java_codeserachnet_process_dataset_with_negative/valid_with_negative.jsonl --test_data_file=/home/wanyao/huangxin/graphcodebert-cpp/dataset/java_codeserachnet_process_dataset_with_negative/test_with_negative.jsonl --codebase_file=/home/wanyao/huangxin/graphcodebert-cpp/dataset/java_codeserachnet_process_dataset_with_negative/codebase_with_negative.jsonl --num_train_epochs 10 --code_length 320 --data_flow_length 64 --nl_length 128 --train_batch_size 64 --eval_batch_size 64 --learning_rate 2e-5 --seed 123456 > /home/wanyao/huangxin/graphcodebert-cpp/txt/train_8.log 2>&1 &
[1] 12414
```

## for trian_9.log to check cpp code search
```shell
(py37-1.7) ➜  graphcodebert-cpp nohup python run.py --output_dir=./saved_models/cpp-without-dfg-triple-loss --config_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base --tokenizer_name=microsoft/graphcodebert-base --lang=java --do_train --train_data_file=/home/wanyao/huangxin/graphcodebert-cpp/dataset/cpp-dataset-split/train_with_negative.jsonl --eval_data_file=/home/wanyao/huangxin/graphcodebert-cpp/dataset/cpp-dataset-split/valid_with_negative.jsonl --test_data_file=/home/wanyao/huangxin/graphcodebert-cpp/dataset/cpp-dataset-split/test_with_negative.jsonl --codebase_file=/home/wanyao/huangxin/graphcodebert-cpp/dataset/cpp-dataset-split/codebase_with_negative.jsonl --num_train_epochs 10 --code_length 320 --data_flow_length 64 --nl_length 128 --train_batch_size 64 --eval_batch_size 64 --learning_rate 2e-5 --seed 123456 > /home/wanyao/huangxin/graphcodebert-cpp/txt/train_9.log 2>&1 &          
[1] 49562
```

### train_10.log for hash code retrieval
```shell
(py37-1.7) ➜  graphcodebert-cpp-hash nohup python run.py --output_dir=./saved_models/cpp-without-dfg-triple-loss-hash --config_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base --tokenizer_name=microsoft/graphcodebert-base --lang=java --do_train --train_data_file=/home/wanyao/huangxin/graphcodebert-cpp-hash/dataset/cpp-dataset-split/train_with_negative.jsonl --eval_data_file=/home/wanyao/huangxin/graphcodebert-cpp-hash/dataset/cpp-dataset-split/valid_with_negative.jsonl --test_data_file=/home/wanyao/huangxin/graphcodebert-cpp-hash/dataset/cpp-dataset-split/test_with_negative.jsonl --codebase_file=/home/wanyao/huangxin/graphcodebert-cpp-hash/dataset/cpp-dataset-split/codebase_with_negative.jsonl --num_train_epochs 10 --code_length 320 --data_flow_length 32 --nl_length 128 --train_batch_size 64 --eval_batch_size 64 --learning_rate 2e-5 --seed 123456 > /home/wanyao/huangxin/graphcodebert-cpp-hash/txt/train_10.log 2>&1 &
[1] 41869
```