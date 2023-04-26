import numpy as np

def get_martix_code_similarity(v1, v2):
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

query = np.random.random((10, 320))
codebase = np.random.random((500, 320))
get_martix_code_similarity(query, codebase)

