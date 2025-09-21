# check_module_map.py
import numpy as np
from pathlib import Path

MODULE_MAP_PATH = r"D:\code\BrainNetworkTransformer-main\source\conf\dataset\group_modules_k7.npy"
MODULE_MAP_PATH = r"D:\code\BrainNetworkTransformer-main\source\conf\dataset\group_modules_k7_zero_based.npy"
SAVE_NORMALIZED_AS = None

def normalize_mods(mods: np.ndarray):
    mods = np.asarray(mods).astype(np.int64).ravel()
    uniq = np.unique(mods)
    K = uniq.size

    # 判定标签方案
    if np.array_equal(uniq, np.arange(K)):              # 0..K-1
        scheme = "zero-based contiguous"
        norm = mods
        remap = {int(u): int(u) for u in uniq}
    elif np.array_equal(uniq, np.arange(1, K + 1)):     # 1..K
        scheme = "one-based contiguous"
        norm = mods - 1
        remap = {int(u): int(u - 1) for u in uniq}
    else:
        scheme = "noncontiguous / gapped"
        # 将排序后的唯一值重映射到 0..K-1
        remap = {int(u): int(i) for i, u in enumerate(uniq)}
        norm = np.vectorize(lambda z: remap[int(z)])(mods)

    return norm.reshape(-1), K, scheme, uniq, remap

def main():
    p = Path(MODULE_MAP_PATH)
    mods = np.load(p).astype(np.int64)
    norm, K, scheme, uniq, remap = normalize_mods(mods)

    print("=== Module Map Check ===")
    print(f"path  : {p}")
    print(f"shape : {mods.shape}  (N={mods.size})")
    print(f"unique: {uniq.tolist()}")
    print(f"K     : {K}")
    print(f"scheme: {scheme}")

    if scheme != "zero-based contiguous":
        print("\n[INFO] 建议统一到 0..K-1。重映射表（old->new）示例：")
        preview = list(remap.items())[:10]
        print(f"  {preview}{' ...' if len(remap)>10 else ''}")

    if SAVE_NORMALIZED_AS:
        out = Path(SAVE_NORMALIZED_AS)
        np.save(out, norm.astype(np.int64))
        print(f"[SAVE] 归一化后的标签写入：{out}")

if __name__ == "__main__":
    main()
