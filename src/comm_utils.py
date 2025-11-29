import io
import numpy as np
import os
import tempfile
import time

def serialized_size_bytes(weights_list):
    bio = io.BytesIO()
    # use numpy.save with allow_pickle True to preserve list-of-arrays ordering
    np.save(bio, np.asarray(weights_list, dtype=object), allow_pickle=True)
    return len(bio.getvalue())

def model_disk_size_bytes(tf_model, tmp_name=None):
    if tmp_name is None:
        tmp_name = tempfile.mktemp(suffix=".keras")
    tf_model.save(tmp_name, overwrite=True)
    size = os.path.getsize(tmp_name)
    try:
        os.remove(tmp_name)
    except Exception:
        pass
    return size

def human_readable_bytes(n):
    for unit in ['B','KB','MB','GB']:
        if n < 1024.0:
            return f"{n:3.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}TB"

def time_inference(model, sample, repeats=50):
    t0 = time.time()
    for _ in range(repeats):
        model.predict(sample, verbose=0)
    t1 = time.time()
    return (t1 - t0) / repeats
