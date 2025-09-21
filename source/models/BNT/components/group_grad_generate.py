# make_group_affinity.py
# 从 abide.npy 的 'corr' 生成群体亲和矩阵，并在群体平均后再做一次 10% 行阈值 + 并集对称化
import os
import math
import numpy as np

# --- 路径 ---
IN_PATH   = r"D:\code\BrainNetworkTransformer-main\source\conf\dataset\abide.npy"
OUT_RAW   = r"D:\code\BrainNetworkTransformer-main\source\conf\dataset\group_affinity.npy"
OUT_TOPK  = r"D:\code\BrainNetworkTransformer-main\source\conf\dataset\group_affinity_topk.npy"

# --- 超参 ---
KEEP_RATIO = 0.10  # 每行保留比例（10%）
EPS = 1e-8         # 防0除

def cosine_affinity_from_corr(R: np.ndarray) -> np.ndarray:
    """R: [N,N] z-FC；返回 A: [N,N] 余弦相似映射到[0,1]，diag=0（未稀疏/对称化）"""
    R = R.astype(np.float32, copy=True)
    np.fill_diagonal(R, 0.0)

    norms = np.linalg.norm(R, axis=1)
    norms = np.maximum(norms, EPS)
    dots = R @ R.T
    cos = dots / (norms[:, None] * norms[None, :])
    cos = np.clip(cos, -1.0, 1.0)
    A = 0.5 * (1.0 + cos)
    np.fill_diagonal(A, 0.0)
    return A

def row_topk_sparsify(A: np.ndarray, keep_ratio: float) -> np.ndarray:
    """对每行仅保留 top-k，其余置0；k=ceil(ratio*(N-1))，排除自环。"""
    N = A.shape[0]
    k = int(math.ceil(max(0.0, keep_ratio) * (N - 1)))
    if k <= 0:
        out = np.zeros_like(A, dtype=np.float32)
        np.fill_diagonal(out, 0.0)
        return out

    A_ = A.copy()
    # 自环置成极小，避免被选入
    np.fill_diagonal(A_, -1.0)

    # 选择每行 top-k 的列索引（不排序的 top-k）
    idx_topk = np.argpartition(A_, kth=A_.shape[1] - k, axis=1)[:, -k:]  # [N,k]

    mask = np.zeros_like(A_, dtype=bool)
    rows = np.arange(N)[:, None]
    mask[rows, idx_topk] = True
    out = np.where(mask, A, 0.0).astype(np.float32, copy=False)

    np.fill_diagonal(out, 0.0)
    return out

def process_subject(R: np.ndarray, keep_ratio: float) -> np.ndarray:
    """单受试者：余弦→[0,1]→行10%→并集对称化→diag=0"""
    A = cosine_affinity_from_corr(R)
    A_topk = row_topk_sparsify(A, keep_ratio)
    A_sym = np.maximum(A_topk, A_topk.T)
    np.fill_diagonal(A_sym, 0.0)
    return A_sym.astype(np.float32, copy=False)

def main():
    print(f"[INFO] Loading {IN_PATH}")
    obj = np.load(IN_PATH, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        obj = obj.item()
    elif isinstance(obj, dict):
        pass
    else:
        raise ValueError("Unsupported npy format: expected 0-d object array or dict with key 'corr'.")

    if "corr" not in obj:
        raise KeyError("The input npy must contain key 'corr'")

    corr = obj["corr"].astype(np.float32)  # [S,N,N]
    if corr.ndim != 3 or corr.shape[1] != corr.shape[2]:
        raise ValueError(f"'corr' must be [S,N,N], got shape={corr.shape}")

    S, N, _ = corr.shape
    print(f"[INFO] Found corr with shape: S={S}, N={N}")
    print(f"[INFO] KEEP_RATIO={KEEP_RATIO} => per-row k=ceil({KEEP_RATIO}*(N-1))={math.ceil(KEEP_RATIO*(N-1))}")

    # ---- 个体 → 群体平均（未二次稀疏）----
    acc = np.zeros((N, N), dtype=np.float64)
    for s in range(S):
        A_s = process_subject(corr[s], KEEP_RATIO)  # [N,N]
        acc += A_s.astype(np.float64)
        if (s + 1) % 50 == 0 or s == 0:
            nnz = np.count_nonzero(A_s)
            density = nnz / (N * N)
            print(f"[INFO] subject {s+1}/{S}: nnz={nnz} ({density:.4%})")

    barA = (acc / float(S)).astype(np.float32)
    # 规范：对称、非负、diag=0（理论上已满足）
    barA = np.maximum(barA, barA.T)
    np.clip(barA, 0.0, 1.0, out=barA)
    np.fill_diagonal(barA, 0.0)

    # 保存未二次稀疏的群体亲和
    os.makedirs(os.path.dirname(OUT_RAW), exist_ok=True)
    np.save(OUT_RAW, barA)
    nnz = np.count_nonzero(barA); density = nnz / (N * N)
    print(f"[SAVE] RAW group affinity -> {OUT_RAW}")
    print(f"[STATS RAW] nnz={nnz} ({density:.4%}), min={barA.min():.6f}, max={barA.max():.6f}, mean={barA.mean():.6f}")

    # ---- 追加步骤：对群体模板再做一次 10% 行阈值 + 并集对称化 ----
    barA_topk = row_topk_sparsify(barA, KEEP_RATIO)
    barA_topk = np.maximum(barA_topk, barA_topk.T)
    np.fill_diagonal(barA_topk, 0.0)

    np.save(OUT_TOPK, barA_topk.astype(np.float32, copy=False))
    nnz2 = np.count_nonzero(barA_topk); density2 = nnz2 / (N * N)
    print(f"[SAVE] POST-TOPK group affinity -> {OUT_TOPK}")
    print(f"[STATS TOPK] nnz={nnz2} ({density2:.4%}), min={barA_topk.min():.6f}, max={barA_topk.max():.6f}, mean={barA_topk.mean():.6f}")

if __name__ == "__main__":
    main()
