import torch


def prepare_targets(lig, bs=3, num_hidden=12):
    """prepare input ligand by append hidden, air, and alive channels
    channel 0: air
    channel 1-n: atom channels | dialated edesp
    channel n+1: alive
    channel n+2-: hidden channels

    Args:
        lig (np.array | torch.tensor): voxel representation of ligand with shape (num_atom_types, x, y, z)
        bs (int, optional): Batchsize. Defaults to 3.

    Returns:
        torch.tensor: tensor with shape (bs, num_atom_types + 3 + num_hidden, x, y, z)
    """

    targets = []
    for _ in range(bs):
        target = torch.tensor(lig)
        is_air = torch.max(target, dim=0)[0].unsqueeze(0)

        n_hidden = num_hidden
        hidden_channels = torch.ones(n_hidden, *target[0].shape)

        targets.append(
            torch.cat(
                [(is_air == 0), target, (is_air == 1), hidden_channels], dim=0
            )
            .unsqueeze(0)
            .to(torch.float)
        )
    return torch.cat(targets, dim=0)


def prepare_inputs(targets, seed_coords: list, num_categories: int = 8):
    """Place a single seed in the center of the input tensor in the alive + hidden channels
    """

    inputs = []
    for _ in range(targets.shape[0]):
        input = torch.zeros_like(targets[0:1]).to(torch.float)

        input[:, num_categories:, seed_coords[0], seed_coords[1], seed_coords[2]] = 1.0
        inputs.append(input)
    return torch.cat(inputs, dim=0)