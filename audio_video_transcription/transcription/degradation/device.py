# Resolve the initial compute device while leaving model imports lazy.
def resolve_device(requested):
    if requested in {"cpu", "cuda"}:
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
