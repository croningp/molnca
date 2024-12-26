import torch


def save_tensor(tensor, filename):
    torch.save(tensor, filename)


def numpy_to_torch_sparse(np_voxel_grid):
    return torch.from_numpy(np_voxel_grid).to_sparse()
