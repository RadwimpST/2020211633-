# compute_modularity.py
# 计算群体模板的加权模块度 Q（Newman–Girvan, weighted, undirected）
# 默认输入:
#   - group_affinity_topk.npy  (N×N, float32)
#   - group_modules_k7.npy     (N, int; 1..K)
# 另：也会对 group_affinity.npy 计算一遍，便于对比。

import numpy as np
import os

AFF_TOPK = r"D:\code\BrainNetworkTransformer-main\source\conf\dataset\group_affinity_topk.npy"
AFF_RAW  = r"D:\code\BrainNetworkTransformer-main\source\conf\dataset\group_affinity.npy"
LABELS   = r"D:\code\BrainNetworkTransformer-main\source\conf\dataset\group_modules_k7.npy"

GAMMA = 1.0   # 分辨率参数（=1 为标准模块度）

def modularity_weighted(A: np.ndarray, labels: np.ndarray, gamma: float = 1.0) -> dict:
    """
    A: [N,N] 加权无向邻接（非负，最好对称，对角0）
    labels: [N] 社区标签（任意整数）
    gamma: 分辨率参数
    返回: {Q, m, strengths, sizes, Q_parts}
    """
    A = np.array(A, dtype=np.float64, copy=True)
    # 对称化 + 去自环（保险）
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    if (A < 0).any():
        raise ValueError("A contains negative weights; expected nonnegative.")

    labels = np.asarray(labels)
    if A.shape[0] != labels.shape[0]:
        raise ValueError(f"Size mismatch: A is {A.shape[0]} but labels is {labels.shape[0]}")

    # strength 和 2m
    k = A.sum(axis=1)                 # 节点强度
    two_m = k.sum()                   # 2m = sum_ij A_ij
    if two_m <= 0:
        return {"Q": 0.0, "m": 0.0, "strengths": k, "sizes": {}, "Q_parts": {}}
    m = two_m / 2.0

    # 按社区累加
    Q_num = 0.0
    sizes = {}
    Q_parts = {}
    for c in np.unique(labels):
        idx = (labels == c)
        sizes[int(c)] = int(idx.sum())
        if sizes[int(c)] == 0:
            Q_parts[int(c)] = 0.0
            continue
        A_cc = A[np.ix_(idx, idx)]
        sumA = A_cc.sum()             # \sum_{i in c} \sum_{j in c} A_ij
        sumk = k[idx].sum()           # \sum_{i in c} k_i
        part = (sumA - gamma * (sumk * sumk) / (2.0 * m))
        Q_num += part
        Q_parts[int(c)] = float(part / (2.0 * m))  # 该社区对 Q 的贡献
    Q = Q_num / (2.0 * m)

    return {"Q": float(Q), "m": float(m), "strengths": k, "sizes": sizes, "Q_parts": Q_parts}

def run_one(path_aff: str, labels: np.ndarray, name: str):
    A = np.load(path_aff).astype(np.float32)
    print(f"\n[INFO] {name}: shape={A.shape}, min={A.min():.6g}, max={A.max():.6g}, mean={A.mean():.6g}")
    nnz = int(np.count_nonzero(A))
    density = nnz / (A.size)
    print(f"[INFO] {name}: nnz={nnz} ({density:.2%})")

    res = modularity_weighted(A, labels, gamma=GAMMA)
    print(f"[RESULT] {name}: Q={res['Q']:.6f} (gamma={GAMMA})")
    print(f"[RESULT] total weight m={res['m']:.6f}")
    print(f"[RESULT] community sizes: {res['sizes']}")
    # 可选：打印每个社区对 Q 的贡献
    parts_str = ", ".join([f"{c}:{res['Q_parts'][c]:.6f}" for c in sorted(res['Q_parts'])])
    print(f"[RESULT] Q parts by community: {parts_str}")

def main():
    labels = np.load(LABELS)
    # 统一到 int（1..K 或 0..K-1 都可）
    labels = labels.astype(int)

    if os.path.exists(AFF_TOPK):
        run_one(AFF_TOPK, labels, name="group_affinity_topk")
    else:
        print(f"[WARN] {AFF_TOPK} not found; skip.")

    if os.path.exists(AFF_RAW):
        run_one(AFF_RAW, labels, name="group_affinity_raw")
    else:
        print(f"[WARN] {AFF_RAW} not found; skip.")

if __name__ == "__main__":
    main()
