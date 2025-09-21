# make_gradients_and_modules.py
# 读取 group_affinity_topk.npy -> 计算 Diffusion Maps 梯度(d=3, alpha=0.5, t=1) -> KMeans(k=7)
# 产物：
#   - group_gradients_d3.npy: [N,3] 的群体梯度坐标 (z-score, 方向已固定)
#   - group_modules_k7.npy : [N] 的簇标签(1..7)
# 依赖：numpy, scikit-learn（无需 SciPy）

import os
import numpy as np
from sklearn.cluster import KMeans

# ---------- 路径 ----------
IN_AFF   = r"D:\code\BrainNetworkTransformer-main\source\conf\dataset\group_affinity_topk.npy"
OUT_GRAD = r"D:\code\BrainNetworkTransformer-main\source\conf\dataset\group_gradients_d3.npy"
OUT_K7   = r"D:\code\BrainNetworkTransformer-main\source\conf\dataset\group_modules_k7.npy"

# ---------- 超参 ----------
ALPHA = 0.5      # Diffusion Maps 的 alpha 归一化
T     = 1        # diffusion time
DIMS  = 3        # 取前3个非平凡梯度
K     = 7        # k-means 簇数
N_INIT = 50      # k-means 重启次数
SEED   = 42      # 随机种子
EPS    = 1e-12   # 数值稳定

def diffusion_maps(K_aff: np.ndarray, alpha: float = 0.5, t: int = 1, dims: int = 3):
    """
    对称非负的亲和矩阵 K_aff ∈ R^{N×N} 上进行 Diffusion Maps。
    返回:
      G ∈ R^{N×dims}  (每列 z-score、方向按“最大绝对值为正”规则固定)
      lambdas: 取到的 dims 个特征值 (对应非平凡模态，降序)
    说明:
      采用经典定义:
        K_alpha = Q^{-alpha} K Q^{-alpha},  Q = diag(K 1)
        D       = diag(K_alpha 1)
        S       = D^{-1/2} K_alpha D^{-1/2}  (对称)
        eigh(S) 得到 (λ, u)，跳过最大平凡特征向量, 取接下来的 dims 个
        右特征向量ψ = D^{-1/2} u，对应 P = D^{-1} K_alpha
        diffusion 坐标: ψ_j * λ_j^t
    """
    K_aff = np.asarray(K_aff, dtype=np.float64)
    N = K_aff.shape[0]
    # 基本校验
    if K_aff.ndim != 2 or K_aff.shape[0] != K_aff.shape[1]:
        raise ValueError(f"K_aff must be square, got {K_aff.shape}")
    # 对称化 + 去自环（保险起见）
    K_aff = 0.5 * (K_aff + K_aff.T)
    np.fill_diagonal(K_aff, 0.0)
    # 确保非负
    K_aff = np.clip(K_aff, 0.0, None, out=K_aff)

    # --- alpha-normalization ---
    q = np.maximum(K_aff.sum(axis=1), EPS)             # [N]
    inv_q_alpha = np.power(q, -alpha)                  # [N]
    K_alpha = (inv_q_alpha[:, None] * K_aff) * inv_q_alpha[None, :]  # [N,N]

    # --- 构造对称算子 S = D^{-1/2} K_alpha D^{-1/2} ---
    d = np.maximum(K_alpha.sum(axis=1), EPS)           # [N]
    inv_sqrt_d = 1.0 / np.sqrt(d)
    S = (inv_sqrt_d[:, None] * K_alpha) * inv_sqrt_d[None, :]

    # --- 特征分解（S 对称，取所有特征值向量）---
    # N=200，直接 eigh 即可
    w, U = np.linalg.eigh(S)  # w 升序, U 的列为特征向量
    # 按特征值降序排序
    idx_desc = np.argsort(w)[::-1]
    w = w[idx_desc]
    U = U[:, idx_desc]

    # 跳过第一个平凡模态 (λ≈1)
    # 寻找第一个接近 1 的特征值的个数（极少情况下>1个）
    # 这里简单取 idx=1..dims
    if len(w) <= (1 + dims):
        raise RuntimeError(f"Not enough nontrivial eigenpairs: got {len(w)} total.")
    lambdas = w[1:1 + dims]
    U_sel   = U[:, 1:1 + dims]   # [N, dims]

    # 右特征向量 ψ = D^{-1/2} U_sel
    psi = (inv_sqrt_d[:, None] * U_sel)  # [N, dims]

    # diffusion 坐标: 乘 λ^t
    if t != 0:
        psi = psi * (lambdas[None, :] ** t)

    # 每列 z-score，并固定方向（令具有最大|值|的那个元素为正）
    G = psi.copy()
    for j in range(G.shape[1]):
        col = G[:, j]
        mu, sd = float(col.mean()), float(col.std(ddof=0))
        col = (col - mu) / (sd + EPS)
        # 方向固定：若 |max| 对应值为负，则翻转
        imax = int(np.argmax(np.abs(col)))
        if col[imax] < 0:
            col = -col
        G[:, j] = col

    return G.astype(np.float32), lambdas.astype(np.float64)

def main():
    print(f"[INFO] Loading affinity: {IN_AFF}")
    A = np.load(IN_AFF).astype(np.float32)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Affinity must be [N,N], got {A.shape}")
    N = A.shape[0]
    print(f"[INFO] A shape={A.shape}, min={A.min():.6g}, max={A.max():.6g}, mean={A.mean():.6g}")

    # --- 计算梯度（d=3）---
    G, lam = diffusion_maps(A, alpha=ALPHA, t=T, dims=DIMS)
    print(f"[INFO] eigenvalues (nontrivial, top {DIMS}): {lam}")
    print(f"[INFO] Gradients G shape={G.shape}, per-dim mean≈0, std≈1 (z-scored)")

    # 保存梯度
    os.makedirs(os.path.dirname(OUT_GRAD), exist_ok=True)
    np.save(OUT_GRAD, G)
    print(f"[SAVE] gradients -> {OUT_GRAD}")

    # --- k-means 聚类（k=7）---
    km = KMeans(n_clusters=K, n_init=N_INIT, max_iter=1000, random_state=SEED)
    labels0 = km.fit_predict(G)   # 0..K-1
    labels1 = (labels0 + 1).astype(np.int32)  # 1..K
    np.save(OUT_K7, labels1)
    print(f"[SAVE] modules (k={K}) -> {OUT_K7}")
    # 简要统计
    counts = np.bincount(labels1, minlength=K+1)[1:]
    print(f"[STATS] cluster counts: {counts.tolist()} (sum={counts.sum()}, N={N})")

if __name__ == "__main__":
    main()
