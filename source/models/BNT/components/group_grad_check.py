# check_connectivity.py
# 对 group_affinity.npy 做检测，包括 连通分量个数、各分量规模、是否对称/对角为零、孤立点统计等
import numpy as np
from collections import deque



# IN_PATH = r"D:\code\BrainNetworkTransformer-main\source\conf\dataset\group_affinity.npy"
IN_PATH = r"D:\code\BrainNetworkTransformer-main\source\conf\dataset\group_affinity_topk.npy"
EPS = 0.0  # 边阈值：>EPS 视为存在边；若想忽略极小噪声，可设为 1e-6

def connected_components_bool(adj_bool: np.ndarray):
    """返回连通分量列表，每个是节点索引的 ndarray。"""
    n = adj_bool.shape[0]
    seen = np.zeros(n, dtype=bool)
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        q = deque([s])
        seen[s] = True
        comp = []
        while q:
            u = q.popleft()
            comp.append(u)
            # 邻居：adj_bool[u] 为 True 的索引
            nbrs = np.flatnonzero(adj_bool[u])
            for v in nbrs:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        comps.append(np.array(comp, dtype=int))
    return comps

def main():
    A = np.load(IN_PATH).astype(np.float32)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"expect square matrix, got {A.shape}")

    n = A.shape[0]
    print(f"[INFO] Loaded group affinity: shape={A.shape}, dtype={A.dtype}")
    print(f"[INFO] Stats: min={A.min():.6g}, max={A.max():.6g}, mean={A.mean():.6g}")
    # === 新增：非零元素的平均值 ===
    mask_nz = (A > EPS)
    np.fill_diagonal(mask_nz, False)  # 不计主对角
    nz_vals = A[mask_nz]
    nz_mean = float(nz_vals.mean()) if nz_vals.size else 0.0
    print(f"[NONZERO] count={nz_vals.size}, mean(nonzero)={nz_mean:.6f}")
    # 基本校验
    sym_ok = np.allclose(A, A.T, atol=1e-6)
    diag_zero = np.allclose(np.diag(A), 0.0, atol=1e-8)
    print(f"[CHECK] symmetric? {sym_ok}")
    print(f"[CHECK] diag==0 ? {diag_zero}")

    # 二值化邻接（无向）
    adj = (A > EPS)
    np.fill_diagonal(adj, False)
    # 强制无向：并集对称化
    adj = np.logical_or(adj, adj.T)

    deg = adj.sum(axis=1).astype(int)
    iso = np.flatnonzero(deg == 0)
    print(f"[DEGREE] min={deg.min()}, max={deg.max()}, mean={deg.mean():.2f}")
    print(f"[ISOLATED] count={iso.size}" + (f", nodes={iso.tolist()}" if iso.size and iso.size <= 20 else ""))

    comps = connected_components_bool(adj)
    sizes = sorted([len(c) for c in comps], reverse=True)
    print(f"[CC] #components = {len(comps)}")
    print(f"[CC] component sizes (desc): {sizes[:10]}" + (" ..." if len(sizes) > 10 else ""))

    if len(comps) > 1:
        # 展示最小的几个小分量（最多 5 个），便于诊断
        comps_sorted = sorted(comps, key=lambda c: len(c))
        show = comps_sorted[: min(5, len(comps_sorted))]
        for i, comp in enumerate(show, 1):
            print(f"[CC] small component {i}: size={len(comp)}, nodes={comp.tolist() if len(comp)<=40 else comp[:40].tolist()+['...']}")

if __name__ == "__main__":
    main()
