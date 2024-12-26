import torch
from torch import nn
from torch.autograd import Variable


def MSE_loss(output, target):
    return Variable(
        torch.sum((output - target) ** 2) / output.data.nelement(), requires_grad=True
    )


def IOU_Loss(out, targets, alife_threshold=0.0):
    out.clamp_(0, 1)
    targets.clamp_(0, 1)

    intersection_mask = (targets > alife_threshold) & (out > alife_threshold)
    intersect = torch.sum(
        torch.clamp(out * intersection_mask +
                    targets * intersection_mask, 0, 1)
    )
    union = torch.sum(torch.clamp(
        (out > alife_threshold).float() + targets, 0, 1))
    o = (union - intersect) / (union + 1e-8)
    return o


def BCE_loss(output, target):
    return Variable(
        torch.nn.functional.BCEWithLogitsLoss(output, target), requires_grad=True
    )


def AIR_Loss(out, target):
    out = torch.clamp(out.to(torch.int), 0, 1)
    target = torch.clamp(target.to(torch.int), 0, 1)

    false_active = torch.sum(out & ~target).float()
    # print(false_active, target.sum())
    return false_active / (target.sum() + 1e-8)


def from_pocket_loss(losses, out, target, pocket_mask, num_categories):
    i_loss = IOU_Loss(out[:, 1:num_categories], target)

    i_loss_ligand_only = IOU_Loss(
        out[:, 1:num_categories] * pocket_mask,
        target * pocket_mask,
    )

    mse_loss = (
        nn.MSELoss(reduction="sum")(out[:, 1:num_categories], target)
        / target.sum()
    )
    mse_loss_ligand_only = (
        nn.MSELoss(reduction="sum")(
            out[:, 1:num_categories] * pocket_mask,
            target * pocket_mask,
        )
        / (target * pocket_mask).sum()
    )

    loss = (
        mse_loss
        + i_loss_ligand_only
        + i_loss
        + mse_loss_ligand_only
    )
    losses["loss"].append(loss.item())
    losses["mse"].append(mse_loss.item())
    losses["iou"].append(i_loss.item())
    losses["mse_ligand_only"].append(mse_loss_ligand_only.item())
    losses["iou_ligand_only"].append(i_loss_ligand_only.item())
    return losses, loss


def from_seed_loss(losses, out, target, num_categories):
    match type(out):
        case torch.Tensor:
            i_loss = IOU_Loss(
                out[:, 1:num_categories], target[:, 1:num_categories]
            )

            mse_loss = (
                nn.MSELoss(reduction="sum")(
                    out[:, 1:num_categories],
                    target[:, 1:num_categories],
                )
                / target[:, 1:num_categories].sum()
            )    
        
            loss = mse_loss + i_loss
            losses["loss"].append(loss.item())
            losses["mse"].append(mse_loss.item())
            losses["iou"].append(i_loss.item())
            return losses, loss
        case list:
            print("CASE LIST")
            for i, o in enumerate(out):
                with torch.no_grad():
                    i_loss = IOU_Loss(
                        o[:, 1:num_categories], target[i][:, 1:num_categories]
                    )
                    mse_loss = (
                        nn.MSELoss(reduction="sum")(
                            o[:, 1:num_categories],
                            target[i][:, 1:num_categories],
                        )
                        / target[i][:, 1:num_categories].sum()
                    )
                    loss = mse_loss + i_loss
                    losses.append(loss.item())
            return losses, loss