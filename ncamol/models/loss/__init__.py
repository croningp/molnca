from .loss_functions import (AIR_Loss, IOU_Loss, MSE_loss, from_pocket_loss,
                             from_seed_loss)

loss_functions = {
    "air": AIR_Loss,
    "mse": MSE_loss,
    "iou": IOU_Loss,
}