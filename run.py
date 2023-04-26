# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import argparse
import logging
import math
import os
import pickle
import random
import traceback

import torch
import json
import faiss
import numpy as np
import torch.nn.functional as F
from model import Model, Second_Stage_Model
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

logger = logging.getLogger(__name__)
from tqdm import tqdm, trange
from datetime import datetime
import multiprocessing

cpu_cont = 16
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 anchor_code_tokens,
                 anchor_code_ids,
                 anchor_position_idx,
                 negative_code_tokens,
                 negative_code_ids,
                 negative_position_idx,
                 nl_tokens,
                 nl_ids,
                 url,
                 ):
        self.anchor_code_tokens = anchor_code_tokens
        self.anchor_code_ids = anchor_code_ids
        self.anchor_position_idx = anchor_position_idx

        self.negative_code_tokens = negative_code_tokens
        self.negative_code_ids = negative_code_ids
        self.negative_position_idx = negative_position_idx

        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url

def convert_examples_to_features(item):
    js, tokenizer, args = item
    # anchor_code
    # anchor_code = js['original_string']
    anchor_code = js['function']
    anchor_code_tokens = tokenizer.tokenize(anchor_code)[:args.code_length - 2]
    anchor_code_tokens = [tokenizer.cls_token] + anchor_code_tokens + [tokenizer.sep_token]
    anchor_code_ids = tokenizer.convert_tokens_to_ids(anchor_code_tokens)
    padding_length = args.code_length - len(anchor_code_ids)
    anchor_code_ids += [tokenizer.pad_token_id] * padding_length
    # position
    anchor_position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(anchor_code_tokens))]
    anchor_position_idx += [tokenizer.pad_token_id] * padding_length

    # code
    # negative_code = js['negative_original_string']
    negative_code = js['negative_function']
    negative_code_tokens = tokenizer.tokenize(negative_code)[:args.code_length - 2]
    negative_code_tokens = [tokenizer.cls_token] + negative_code_tokens + [tokenizer.sep_token]
    negative_code_ids = tokenizer.convert_tokens_to_ids(negative_code_tokens)
    padding_length = args.code_length - len(negative_code_ids)
    negative_code_ids += [tokenizer.pad_token_id] * padding_length
    # position
    negative_position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(negative_code_tokens))]
    negative_position_idx += [tokenizer.pad_token_id] * padding_length

    # nl
    # nl = ' '.join(js['docstring_tokens'])
    nl = js['docstring']
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length - 2]
    nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(anchor_code_tokens, anchor_code_ids, anchor_position_idx,
                         negative_code_tokens, negative_code_ids, negative_position_idx,
                         nl_tokens, nl_ids, js['url'])

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, pool=None):
        self.args = args
        prefix = file_path.split('/')[-1][:-6]
        cache_file = args.output_dir + '/' + prefix + '.pkl'
        if os.path.exists(cache_file):
            self.examples = pickle.load(open(cache_file, 'rb'))
        else:
            self.examples = []
            data = []
            with open(file_path) as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    js = json.loads(line)
                    data.append((js, tokenizer, args))
                    # if idx >= 999:
                    #     break
            self.examples = pool.map(convert_examples_to_features, tqdm(data, total=len(data)))
            pickle.dump(self.examples, open(cache_file, 'wb'))

        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("anchor_code_tokens: {}".format([x.replace('\u0120', '_') for x in example.anchor_code_tokens]))
                logger.info("anchor_code_ids: {}".format(' '.join(map(str, example.anchor_code_ids))))
                logger.info("anchor_position_idx: {}".format(example.anchor_position_idx))
                logger.info("negative_code_tokens: {}".format([x.replace('\u0120', '_') for x in example.negative_code_tokens]))
                logger.info("negative_code_ids: {}".format(' '.join(map(str, example.negative_code_ids))))
                logger.info("negative_position_idx: {}".format(example.negative_position_idx))
                logger.info("nl_tokens: {}".format([x.replace('\u0120', '_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # anchor attn_mask
        # calculate graph-guided masked function
        anchor_attn_mask = np.zeros((self.args.code_length,
                                     self.args.code_length), dtype=bool)
        # calculate begin index of node and max length of input
        anchor_node_index = sum([i > 1 for i in self.examples[item].anchor_position_idx])
        anchor_max_length = sum([i != 1 for i in self.examples[item].anchor_position_idx])
        # sequence can attend to sequence
        anchor_attn_mask[:anchor_node_index, :anchor_node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].anchor_code_ids):
            if i in [0, 2]:
                anchor_attn_mask[idx, :anchor_max_length] = True

        # negative attn_mask
        # calculate graph-guided masked function
        negative_attn_mask = np.zeros((self.args.code_length,
                                       self.args.code_length), dtype=bool)
        # calculate begin index of node and max length of input
        negative_node_index = sum([i > 1 for i in self.examples[item].negative_position_idx])
        negative_max_length = sum([i != 1 for i in self.examples[item].negative_position_idx])
        # sequence can attend to sequence
        negative_attn_mask[:negative_node_index, :negative_node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].negative_code_ids):
            if i in [0, 2]:
                negative_attn_mask[idx, :negative_max_length] = True

        return (torch.tensor(self.examples[item].anchor_code_ids),
                torch.tensor(anchor_attn_mask),
                torch.tensor(self.examples[item].anchor_position_idx),
                torch.tensor(self.examples[item].negative_code_ids),
                torch.tensor(negative_attn_mask),
                torch.tensor(self.examples[item].negative_position_idx),
                torch.tensor(self.examples[item].nl_ids))


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, tokenizer, pool, second_stage_model):
    """ Train the model """
    # get training dataset
    train_dataset = TextDataset(tokenizer, args, args.train_data_file, pool)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)

    # get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_dataloader) * args.num_train_epochs)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader) * args.num_train_epochs)

    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()

    model.train()
    tr_num, tr_loss, best_mrr = 0, 0, 0
    for idx in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # get inputs
            anchor_code_inputs = batch[0].to(args.device)
            anchor_attn_mask = batch[1].to(args.device)
            anchor_position_idx = batch[2].to(args.device)

            negative_code_inputs = batch[3].to(args.device)
            negative_attn_mask = batch[4].to(args.device)
            negative_position_idx = batch[5].to(args.device)

            nl_inputs = batch[6].to(args.device)
            # get code and nl vectors
            hash_scale = math.pow((1.0 * step + 1.0), 0.5)
            anchor_code_vec = model(code_inputs=anchor_code_inputs, attn_mask=anchor_attn_mask, position_idx=anchor_position_idx, hash_scale=hash_scale, retrieval_model=False)
            negative_code_vec = model(code_inputs=negative_code_inputs, attn_mask=negative_attn_mask, position_idx=negative_position_idx, hash_scale=hash_scale, retrieval_model=False)
            nl_vec = model(nl_inputs=nl_inputs, hash_scale=hash_scale, retrieval_model=False)

            anchor_cos_sim = F.cosine_similarity(nl_vec, anchor_code_vec, dim=1).clamp(min=-1, max=1)
            neg_cos_sim = F.cosine_similarity(nl_vec, negative_code_vec, dim=1).clamp(min=-1, max=1)
            loss = (0.5 - anchor_cos_sim + neg_cos_sim).clamp(min=1e-6).mean()

            # report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step + 1) % 100 == 0:
                logger.info("epoch {} step {} loss {}".format(idx, step + 1, round(tr_loss / tr_num, 5)))
                tr_loss = 0
                tr_num = 0

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        test_results = evaluate(args, model, tokenizer, args.test_data_file, pool, second_stage_model, hash_scale)
        for test_key, test_value in test_results.items():
            logger.info("--------------------------------------------")
            logger.info("epoch: %s, test %s = %s", idx, test_key, round(test_value, 4))
            logger.info("--------------------------------------------")

        # evaluate
        results = evaluate(args, model, tokenizer, args.eval_data_file, pool, second_stage_model, hash_scale)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value, 4))

        # save best model
        if results['eval_mrr'] > best_mrr:
            best_mrr = results['eval_mrr']
            logger.info("  " + "*" * 20)
            logger.info("  Best mrr:%s", round(best_mrr, 4))
            logger.info("  " + "*" * 20)

            checkpoint_prefix = 'checkpoint-best-mrr'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)


def get_martix_code_similarity(v1, v2):
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

def evaluate(args, model, tokenizer, file_name, pool, second_stage_model, hash_scale):
    query_dataset = TextDataset(tokenizer, args, file_name, pool)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size, num_workers=4)

    code_dataset = TextDataset(tokenizer, args, args.codebase_file, pool)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # multi-gpu evaluate
    # if args.n_gpu > 1 and eval_when_training is False:
    #     model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    # # ---------------------------------------------------
    # self.args = args
    # prefix = file_path.split('/')[-1][:-6]
    # cache_file = args.output_dir + '/' + prefix + '.pkl'
    # if os.path.exists(cache_file):
    #     self.examples = pickle.load(open(cache_file, 'rb'))
    # else:
    #     self.examples = []
    #     data = []
    #     with open(file_path) as f:
    #         for idx, line in enumerate(f):
    #             line = line.strip()
    #             js = json.loads(line)
    #             data.append((js, tokenizer, args))
    #             # if idx >= 999:
    #             #     break
    #     self.examples = pool.map(convert_examples_to_features, tqdm(data, total=len(data)))
    #     pickle.dump(self.examples, open(cache_file, 'wb'))
    # # ---------------------------------------------------

    model.eval()
    code_hash_vecs = []
    code_vecs = []
    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)
        attn_mask = batch[1].to(args.device)
        position_idx = batch[2].to(args.device)
        with torch.no_grad():
            code_hash_vec = model(code_inputs=code_inputs, attn_mask=attn_mask, position_idx=position_idx, hash_scale=hash_scale, retrieval_model=True)
            code_hash_vecs.append(code_hash_vec.cpu().numpy())

            code_ves = second_stage_model(code_inputs=code_inputs, attn_mask=attn_mask, position_idx=position_idx)
            code_vecs.append(code_ves.cpu().numpy())

    code_hash_vecs = np.concatenate(code_hash_vecs, 0)
    code_hash_vecs = (code_hash_vecs > 0).astype(np.uint8)
    code_vecs = np.concatenate(code_vecs, 0)

    index = faiss.IndexBinaryFlat(len(code_hash_vecs[0]) * 8)
    index.add(code_hash_vecs)

    nl_hash_vecs = []
    nl_vecs = []
    query_nl_inputs = []
    for batch in query_dataloader:
        nl_inputs = batch[6].to(args.device)
        with torch.no_grad():
            # nl_hash_vec = model(nl_inputs=nl_inputs, hash_scale=hash_scale, retrieval_model=True)
            # nl_hash_vecs.append(nl_hash_vec.cpu().numpy())
            # nl_vec = second_stage_model(nl_inputs=nl_inputs)
            # nl_vecs.append(nl_vec.cpu().numpy())
            query_nl_inputs.append(nl_inputs)

    model.train()
    # nl_hash_vecs = np.concatenate(nl_hash_vecs, 0)
    # nl_vecs = np.concatenate(nl_vecs, 0)
    query_nl_inputs = torch.cat(query_nl_inputs, 0)

    test_start = datetime.now()

    sort_ids = []
    for nl_input in query_nl_inputs:
        with torch.no_grad():
            nl_hash_vec = model(nl_inputs=nl_input.unsqueeze(0), hash_scale=hash_scale, retrieval_model=True)
            nl_vec = second_stage_model(nl_inputs=nl_input.unsqueeze(0))

            nl_hash_vec = (nl_hash_vec.cpu().numpy() > 0).astype(np.uint8)
            D, I = index.search(nl_hash_vec, 1000)
            candidate_code_vecs = code_vecs[I[0].tolist()]
            scores = get_martix_code_similarity(nl_vec.cpu().numpy(), candidate_code_vecs)
            sort_id = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]
            sort_ids.append(I[0][sort_id[0].tolist()])

    logger.info("--------------------------------------------")
    logger.info(f"Time elapsed for {len(query_nl_inputs)} times code search = {datetime.now() - test_start}")
    logger.info(f"Average time elapsed for per code search = {(datetime.now() - test_start)/len(query_nl_inputs)}")
    logger.info("--------------------------------------------")

    # scores=np.matmul(nl_vecs,code_vecs.T)
    # scores = get_martix_code_similarity(nl_vecs, code_vecs)
    # sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]

    nl_urls = []
    code_urls = []
    for example in query_dataset.examples:
        nl_urls.append(example.url)

    for example in code_dataset.examples:
        code_urls.append(example.url)

    ranks = []
    for url, sort_id in zip(nl_urls, sort_ids):
        rank = 0
        find = False
        for idx in sort_id[:1000]:
            if find is False:
                rank += 1
            if code_urls[idx] == url:
                find = True
        if find:
            ranks.append(1 / rank)
        else:
            ranks.append(0)

    result = {
        "eval_mrr": float(np.mean(ranks))
    }

    return result

def demo_evaluate(args, model, tokenizer, file_name, pool, second_stage_model, hash_scale):
    query_dataset = TextDataset(tokenizer, args, file_name, pool)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size, num_workers=4)

    code_dataset = TextDataset(tokenizer, args, args.codebase_file, pool)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size, num_workers=4)

    nl_urls = []
    code_urls = []
    codebase_code = []
    for example in query_dataset.examples:
        nl_urls.append(example.url)

    for example in code_dataset.examples:
        code_urls.append(example.url)

    codebase_code = {}
    with open("/home/wanyao/huangxin/graphcodebert-cpp-hash/dataset/cpp-dataset-split/codebase_with_negative.jsonl") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            codebase_code[js['url']] = js

    # multi-gpu evaluate
    # if args.n_gpu > 1 and eval_when_training is False:
    #     model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()

    cache_code_hash_vecs = os.path.join(args.output_dir, "cache_code_hash_vecs.pkl")
    cache_code_vecs = os.path.join(args.output_dir, "cache_code_vecs.pkl")
    if os.path.exists(cache_code_hash_vecs) and os.path.exists(cache_code_vecs):
        code_hash_vecs = pickle.load(open(cache_code_hash_vecs, 'rb'))
        code_vecs = pickle.load(open(cache_code_vecs, 'rb'))
        logger.info("code_hash_vecs and code_vecs load done!")
    else:
        code_hash_vecs = []
        code_vecs = []
        for batch in code_dataloader:
            code_inputs = batch[0].to(args.device)
            attn_mask = batch[1].to(args.device)
            position_idx = batch[2].to(args.device)
            with torch.no_grad():
                code_hash_vec = model(code_inputs=code_inputs, attn_mask=attn_mask, position_idx=position_idx,hash_scale=hash_scale, retrieval_model=True)
                code_hash_vecs.append(code_hash_vec.cpu().numpy())

                code_ves = second_stage_model(code_inputs=code_inputs, attn_mask=attn_mask, position_idx=position_idx)
                code_vecs.append(code_ves.cpu().numpy())
        code_hash_vecs = np.concatenate(code_hash_vecs, 0)
        code_hash_vecs = (code_hash_vecs > 0).astype(np.uint8)
        code_vecs = np.concatenate(code_vecs, 0)

        pickle.dump(code_hash_vecs, open(cache_code_hash_vecs, 'wb'))
        pickle.dump(code_vecs, open(cache_code_vecs, 'wb'))
        logger.info("code_hash_vecs and code_vecs dump done!")

    index = faiss.IndexBinaryFlat(len(code_hash_vecs[0]) * 8)
    index.add(code_hash_vecs)

    logger.info("--------------------------------------------")
    logger.info("codebase index construct is finished")
    while True:
        try:
            query = input("Input Query: ")
            n_results = int(input("How many results: "))
            # query = "/ @{\n/ @name BuildDB API"
            # n_results = 10
            # start to time
            test_start = datetime.now()
            query_tokens = tokenizer.tokenize(query)[:args.nl_length - 2]
            query_tokens = [tokenizer.cls_token] + query_tokens + [tokenizer.sep_token]
            query_ids = tokenizer.convert_tokens_to_ids(query_tokens)
            padding_length = args.nl_length - len(query_ids)
            query_ids += [tokenizer.pad_token_id] * padding_length
            query_ids = torch.tensor(query_ids).to(args.device)
        except Exception:
            logger.info("Exception while parsing your input: ")
            traceback.print_exc()
            break

        with torch.no_grad():
            nl_hash_vec = model(nl_inputs=query_ids.unsqueeze(0), hash_scale=hash_scale, retrieval_model=True)
            nl_vec = second_stage_model(nl_inputs=query_ids.unsqueeze(0))

            nl_hash_vec = (nl_hash_vec.cpu().numpy() > 0).astype(np.uint8)
            D, I = index.search(nl_hash_vec, 1000)
            candidate_code_vecs = code_vecs[I[0].tolist()]
            scores = get_martix_code_similarity(nl_vec.cpu().numpy(), candidate_code_vecs)
            sort_id = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]
            top_k = I[0][sort_id[0][:n_results]]
            logger.info("--------------------------------------------")
            print(f"Time elapsed for code search = {datetime.now() - test_start}")
            print(f"search result index: {np.array(code_urls)[top_k]}")
            for i in range(len(np.array(code_urls)[top_k])):
                print(f"function top-{i}, ground truth index-{np.array(code_urls)[top_k][i]}:")
                print(codebase_code[np.array(code_urls)[top_k][i]]['function'])
                print("-------\n")
            logger.info("--------------------------------------------")


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")

    parser.add_argument("--lang", default=None, type=str,
                        help="language.")

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_demo", action='store_true',
                        help="Code Search for Demo.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    pool = multiprocessing.Pool(cpu_cont)

    # print arguments
    args = parser.parse_args()

    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args.seed)

    # build model
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    pre_train_model = RobertaModel.from_pretrained(args.model_name_or_path)
    model = Model(pre_train_model)
    logger.info("Training/evaluation parameters %s", args)
    model.to(args.device)

    # build second stage model for specific dense vector search
    second_stage_model = Second_Stage_Model(pre_train_model)
    second_stage_model_path = '/home/wanyao/huangxin/graphcodebert-cpp/saved_models/cpp-without-dfg-triple-loss/checkpoint-best-mrr/model.bin'
    second_stage_model.load_state_dict(torch.load(second_stage_model_path), strict=False)
    second_stage_model.to(args.device)
    for name, parameter in second_stage_model.named_parameters():
        parameter.requires_grad = False

    # Training
    if args.do_train:
        train(args, model, tokenizer, pool, second_stage_model)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir), strict=False)
        model.to(args.device)
        result = evaluate(args, model, tokenizer, args.eval_data_file, pool, second_stage_model)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir), strict=False)
        model.to(args.device)
        result = evaluate(args, model, tokenizer, args.test_data_file, pool, second_stage_model)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    if args.do_demo:
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir), strict=False)
        model.to(args.device)
        result = demo_evaluate(args, model, tokenizer, args.test_data_file, pool, second_stage_model, 3200)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    return results


if __name__ == "__main__":
    main()
