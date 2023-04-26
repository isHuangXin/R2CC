import argparse
import numpy as np
import os, psutil
import copy
import time
import faiss
# 程序运行时间 = cpu时间 + io时间 + 阻塞时间


def faiss_cos_sim(args):
    np.random.seed(args.seed)
    print('使用Faiss实现余弦相似度搜索-[暴力版]'.center(40, '='), flush=True)
    print('随机生成%d个二值化向量，维度：%d' % (args.total, args.dim), flush=True)
    # numpy生成[a, b)区间的随机数, (b - a) * random_sample() + a
    x_b = 2 * np.random.random((args.total, args.dim)) - 1
    x_b[x_b <= 0] = 0
    x_b[x_b > 0] = 1
    x_b = x_b.astype(np.uint8)

    # for test
    x_q = 2 * np.random.random((args.test_times, args.dim)) - 1
    x_q[x_q <= 0] = 0
    x_q[x_q > 0] = 1
    x_q = x_q.astype(np.uint8)

    index = faiss.IndexBinaryFlat(args.dim * 8)
    print(index.is_trained)
    index.add(x_b)

    process = psutil.Process(os.getpid())
    print('Used Memory:', process.memory_info().rss / 1024 / 1024, 'MB')

    print("单条查询测试".center(40, '-'))
    q = x_q[0]
    q = q[np.newaxis, :]
    start_one = time.time()
    D_one, I_one = index.search(q, args.top_k)
    end_one = time.time()
    print(f"在{args.total}里使用余弦相似度搜索单条数据花费时间: {end_one - start_one}")
    print(I_one)
    print(D_one)

    print('批量查询测试'.center(40, '-'))
    print('批量测试次数：%d 次，请稍候...' % args.test_times)
    start_batch = time.time()
    D, I = index.search(x_q, args.top_k)
    end_batch = time.time()
    total_time = end_batch - start_batch
    print('总用时:%f秒, 平均用时:%f秒' % (total_time, total_time / args.test_times))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='faiss速度测试工具')
    parser.add_argument('--total', default=10000, type=int, help='总数据量,默认10万')  # required=True,
    parser.add_argument('--dim', default=768, type=int, help='向量维度,默认768')
    parser.add_argument('--test_times', default=100, type=int, help='测试次数')
    parser.add_argument('--top_k', default=10, type=int, help='每次返回条数,默认5')
    parser.add_argument('--gpu', default=-1, type=int, help='使用GPU,-1=不使用（默认），0=使用第1个，>0=使用全部')
    parser.add_argument('--seed', default=1234, type=int, help='make reproducible')
    args = parser.parse_args()
    print(args)
    faiss_cos_sim(args)