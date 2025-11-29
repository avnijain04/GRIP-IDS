"""
Compute model footprint: file size (MB), parameter count, and simple CPU inference latency.
Saves results to results/model_footprint.json and prints a table.
"""
import os, sys, time, json
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

MODELS_DIR = _root / "models"
RESULTS = _root / "results"
RESULTS.mkdir(exist_ok=True)

def try_load_keras(model_path):
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        m = load_model(str(model_path))
        return m
    except Exception as e:
        return None

def count_params_from_weights(weights):
    import numpy as np
    total = 0
    for w in weights:
        total += int(w.size)
    return total

def analyze_model(path: Path):
    item = {"path": str(path), "size_bytes": path.stat().st_size}
    model = try_load_keras(path) if path.suffix in (".h5", ".keras", ".hdf5") else None
    if model is not None:
        try:
            item["params"] = int(model.count_params())
        except Exception:
            item["params"] = None
        # measure inference latency on a small batch if possible
        try:
            import numpy as np
            # build a dummy input based on model.input_shape
            shape = model.input_shape
            # shape is (None, seq, features) or (None, features)
            if shape is None:
                item["latency_ms"] = None
            else:
                dummy_shape = [1] + [int(s) if s is not None else 1 for s in shape[1:]]
                arr = np.random.randn(*dummy_shape).astype("float32")
                # warmup
                for _ in range(2):
                    _ = model.predict(arr, batch_size=1)
                t0 = time.time()
                for _ in range(5):
                    _ = model.predict(arr, batch_size=1)
                t1 = time.time()
                item["latency_ms"] = round((t1 - t0) / 5 * 1000, 3)
        except Exception:
            item["latency_ms"] = None
    else:
        # fallback: if it's an npz weights archive, read param counts
        if path.suffix == ".npz":
            try:
                import numpy as np
                data = np.load(str(path))
                total = 0
                for k in data:
                    total += int(data[k].size)
                item["params"] = total
            except Exception:
                item["params"] = None
        else:
            item["params"] = None
            item["latency_ms"] = None
    item["size_mb"] = round(item["size_bytes"] / (1024*1024), 4)
    return item

def main():
    models = list(MODELS_DIR.glob("*"))
    report = []
    for m in models:
        try:
            r = analyze_model(m)
            report.append(r)
            print(f"{m.name:30} size={r['size_mb']}MB params={r.get('params')} latency_ms={r.get('latency_ms')}")
        except Exception as e:
            print("Failed to analyze", m, ":", e)
    outp = RESULTS / "model_footprint.json"
    with open(outp, "w") as f:
        json.dump(report, f, indent=2)
    print("Saved model footprint:", outp)

if __name__ == "__main__":
    main()
