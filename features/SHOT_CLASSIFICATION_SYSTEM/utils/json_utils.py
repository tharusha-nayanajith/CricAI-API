import numpy as np

def to_json_safe(obj):
    """Recursively convert numpy types to JSON-serializable Python types"""
    
    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(to_json_safe(v) for v in obj)

    return obj