import numpy as np, sys, pathlib
p = r"D:\code\BrainNetworkTransformer-main\source\conf\dataset\abide.npy"
obj = np.load(p, allow_pickle=True)
print("Top:", type(obj), getattr(obj, "dtype", None), getattr(obj, "shape", None))
# 如果是0维object数组，取出真正对象
if isinstance(obj, np.ndarray) and obj.dtype==object and obj.shape==():
    obj = obj.item()
    print("Unboxed ->", type(obj))

if isinstance(obj, dict):
    print("Dict keys:", list(obj.keys()))
    for k,v in obj.items():
        if isinstance(v, np.ndarray):
            print(f"- {k}: ndarray shape={v.shape}, dtype={v.dtype}")
            if v.ndim>=2:
                print("  sample stats:", float(np.nanmin(v.ravel()[:10000])),
                      float(np.nanmax(v.ravel()[:10000])), float(np.nanmean(v.ravel()[:10000])))
        else:
            print(f"- {k}: {type(v)} -> {str(v)[:120]}")
elif isinstance(obj, np.ndarray):
    print("Array:", obj.shape, obj.dtype)
else:
    print("Object preview:", str(obj)[:300])