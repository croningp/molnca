def flush_cuda_cache(model):
    import torch
    import gc

    try:
        model.cpu()
        del model
    except:
        ...

    gc.collect()
    torch.cuda.empty_cache()
