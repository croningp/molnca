import lightning as L
import torch
import torch.nn.functional as F

from ncamol.models.loss import from_pocket_loss, from_seed_loss

from .perception_block import PerceptionBlock
from .update_block import UpdateBlock


class LitModel(L.LightningModule):
    def __init__(
        self,
        # loss: torch.nn.Module,
        from_pocket: bool = False,
        lr: float = 1e-4,
        alive_threshold: float = 0.1,  # threshold for cell to be alive
        cell_fire_rate: float = 0.5,  # probability of cell firing / i.e. dropout
        # number of channels in the input (7 atom types + 1 solvent channel)
        num_categories: int = 8,
        num_hidden_channels: int = 12,  # number of channels in the hidden layers
        steps: int = 48,  # number of steps to run the model for
        channel_dims: list[int] = [
            42,
            42,
        ],  # number of channels in the hidden layers
        normal_std: float = 0.02,  # standard deviation for normal initialization
        use_normal_init: bool = True,  # whether to use normal initialization
        zero_bias: bool = True,
        return_alive_mask: bool = True,
        losses: list = ["mse", "iou"],
    ):
        super().__init__()
        self.save_hyperparameters()
        if from_pocket:
            self.from_pocket = from_pocket
        else:
            self.from_pocket = False

        self.lr = lr
        # block types, hidden channels, alive channel
        self.num_channels = (
            num_categories + num_hidden_channels + 1
        )  
        self.steps = steps
        self.num_categories = num_categories
        self.channel_dims = channel_dims
        self.alive_threshold = alive_threshold
        self.cell_fire_rate = cell_fire_rate
        self.return_alive_mask = return_alive_mask
        self.losses = losses

        self.perception = PerceptionBlock(
            num_channels=self.num_channels,
            normal_std=normal_std,
            use_normal_init=use_normal_init,
            zero_bias=zero_bias,
        )

        self.update = UpdateBlock(
            num_channels=self.num_channels,
            channel_dims=channel_dims,
            normal_std=normal_std,
            use_normal_init=use_normal_init,
            zero_bias=zero_bias,
        )

    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def params(self):
        # number of parameters in human readable format
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def lossfct(self, x, target):
        if not self.from_pocket:
            losses = {"mse": [], "iou": [], "loss": []}
            losses, loss = from_seed_loss(losses, x, target, self.num_categories)
        else:
            losses = {
                "loss": [],
                "mse": [],
                "iou": [],
                "mse_ligand_only": [],
                "iou_ligand_only": [],
            }
            losses, loss = from_pocket_loss(                
                    losses=losses,
                    out=x,
                    target=target,
                    pocket_mask=self.pocket_mask,
                    num_categories=self.num_categories
                )
        return losses, loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def forward(
        self, x: torch.Tensor, steps: int = 48, accumulate: bool = False
    ):
        """
        For inference apply the model for a number of steps to the input

        Parameters
        ----------
        x : torch.Tensor
            input tensor of shape (batch_size, num_channels, x, y, z)
        steps : int, optional
            number of steps to run the model for, by default 48
        accumulate : bool, optional
            whether to accumulate the output of each step, by default False

        Returns
        -------
        torch.Tensor
            output tensor of shape (batch_size, steps, num_channels, x, y, z)
        """
        self.eval()
        with torch.no_grad():
            if accumulate:
                traj = [
                    self.validation_step(x, steps=1, return_life_mask=False)
                    for _ in range(steps)
                ]
            else:
                for _ in range(steps):
                    x = self.validation_step(x, steps=1).get("out", None)
                traj = x
        self.train()
        return traj

    def training_step(
        self, x: torch.Tensor, **kwargs
    ):
        steps = kwargs.get("steps", self.steps)
        if isinstance(steps, list):
            steps = torch.randint(steps[0], steps[1], (1,)).item()

        x, target = x
        if self.from_pocket:
            self.pocket_mask = (x[:, 1:self.num_categories] != 1).float()

        # for training
        for _ in range(steps):
            pre_life_mask = self.alive_cells(x) > self.alive_threshold

            out = self.perception(x)
            out = self.update(out)

            # different than in paper growing 3d artefacts
            drop_out_mask = (
                torch.rand_like(x[:, :1, :, :, :]) < self.cell_fire_rate
            )

            out = out * drop_out_mask.float()

            # stochastic update
            x = x + out

            # post life mask
            post_life_mask = self.alive_cells(x) > self.alive_threshold
            life_mask = (pre_life_mask & post_life_mask).float()
            # print(f"Prelife active: {pre_life_mask.sum()}, postlife active: {post_life_mask.sum()}")

            x = x * life_mask
            #  air cells are set to 1
            x[:, :1, :, :, :][life_mask == 0] += torch.tensor(1.0)

        losses, loss = self.lossfct(x, target)

        for key, value in losses.items():
            self.log(key, value[-1])
        self.log("loss", loss)

        target_alive = target[:, self.num_categories : self.num_categories + 1, :, :, :] > 0
        self.log(
            "train_alive",
            life_mask.sum() / target_alive.sum(),
        )

        return_dict = {"loss": loss, "alive_mask": life_mask, "out": x}
        return return_dict

    def validation_step(self, x, steps=48, **kwargs):
        if self.from_pocket:
            self.pocket_mask = (x[:, 1:self.num_categories] != 1).float()

        # for training
        for _ in range(steps):
            pre_life_mask = self.alive_cells(x) > self.alive_threshold

            out = self.perception(x)
            out = self.update(out)

            # different than in paper growing 3d artefacts
            drop_out_mask = (
                torch.rand_like(x[:, :1, :, :, :]) < self.cell_fire_rate
            )

            out = out * drop_out_mask.float()

            # stochastic update
            x = x + out

            # post life mask
            post_life_mask = self.alive_cells(x) > self.alive_threshold
            life_mask = (pre_life_mask & post_life_mask).float()
            # print(f"Prelife active: {pre_life_mask.sum()}, postlife active: {post_life_mask.sum()}")

            x = x * life_mask
            #  air cells are set to 1
            x[:, :1, :, :, :][life_mask == 0] += torch.tensor(1.0)

        return {"alive_mask": life_mask, "out": x}

    def alive_cells(self, x):
        return F.max_pool3d(
            x[:, self.num_categories : self.num_categories + 1, :, :, :],
            kernel_size=3,
            stride=1,
            padding=1,
        )
