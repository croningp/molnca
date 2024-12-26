import lightning as pl
import torch
import torch.nn.functional as F
import os
from ncamol.models.loss import from_seed_loss


from .perception_block import PerceptionBlock
from .update_block import UpdateBlock


class LitModelExtForce(pl.LightningModule):
    def __init__(
        self,
        # loss: torch.nn.Module,
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
    ):  # whether to zero the bias):
        """Class to train a model to conditionally swtich an object between two states
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.num_channels = (
            num_categories + num_hidden_channels + 1
        )  # block types, hidden channels, alive channel
        self.num_hidden_channels = num_hidden_channels
        self.steps = steps
        self.num_categories = num_categories
        self.channel_dims = channel_dims
        self.alive_threshold = alive_threshold
        self.cell_fire_rate = cell_fire_rate
        self.loss = 0 # backward prop after epoch end
        self.automatic_optimization=False # custom opt. at end of epoch
        self.lowest_loss = torch.inf

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

    def lossfct(self, x: torch.Tensor, target: torch.Tensor):
        """Loss fct wrapper. Return combined loss (MSE + IOU) as well as individual for loggging

        Args:
            x (torch.Tensor): predicted state
            target (torch.Tensor): target state

        Returns:
            dict, torch.Tensor: summary of losses and total loss
        """
        losses = {"mse": [], "iou": [], "loss": []}
        losses, loss = from_seed_loss(losses, x, target, self.num_categories)
        return losses, loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return self.optimizer

    def forward(
        self, x: torch.Tensor, steps: int = 48, accumulate: bool = False, state=0,
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
                    self.validation_step(x, steps=1, return_life_mask=False, state=state)
                    for _ in range(steps)
                ]
            else:
                for _ in range(steps):
                    x = self.validation_step(x, steps=1, state=state).get("out", None)
                    traj = x
        self.train()
        return traj

    def training_step(
        self,
        input: dict,
        **kwargs
    ):
        steps = kwargs.get("steps", self.steps)
        if isinstance(steps, list):
            steps = torch.randint(steps[0], steps[1], (1,)).item()

        x = input["input"]
        target = input["target"]
        state = input["state"]

        switch = self.light_state(state=state, shape=x.shape[-3:]).to(x.device)

        for _ in range(steps):
            pre_life_mask = self.alive_cells(x) > self.alive_threshold

            # apply light state
            x[:, -self.num_hidden_channels :, ...] = (
                x[:, -self.num_hidden_channels :, ...] * switch
            )
            out = self.perception(x)
            out = self.update(out)
            # drop out mask
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

            x = x * life_mask
            #  air cells are set to 1
            x[:, :1, :, :, :][life_mask == 0] += torch.tensor(1.0)

        losses, loss = self.lossfct(x, target)
        # losses for logging only
        for key, value in losses.items():
            self.log(key, value[-1])
        self.loss += loss

        target_alive = target[:, self.num_categories : self.num_categories + 1, :, :, :] > 0
        self.log(
            "train_alive",
            life_mask.sum() / target_alive.sum(),
        )

        return_dict = {"loss": loss, "alive_mask": life_mask, "out": x}
        return return_dict

    def on_train_epoch_end(self):
        # accumulated grads of all states
        self.optimizer.zero_grad()
        self.loss.backward()

        self.clip_gradients(self.optimizer, gradient_clip_val=3, gradient_clip_algorithm="norm")
        self.optimizer.step()

        if not self.automatic_optimization and self.loss < self.lowest_loss:
            ckpt_path = os.path.join(self.trainer.log_dir, 'checkpoints', 'ckpt.pt')
            self.trainer.save_checkpoint(ckpt_path)

            self.lowest_loss = self.loss.detach()
        self.loss = 0

        return None

    def validation_step(self, x, steps=48, state=0, **kwargs):

        switch = self.light_state(state=state, shape=x.shape[-3:]).to(x.device)
        # for training
        for _ in range(steps):
            pre_life_mask = self.alive_cells(x) > self.alive_threshold

            # apply light state
            x[:, -self.num_hidden_channels :, ...] = (
                x[:, -self.num_hidden_channels :, ...] * switch
            )

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

            x = x * life_mask
            #  air cells are set to 1
            x[:, :1, :, :, :][life_mask == 0] += torch.tensor(1.0)

        return {"alive_mask": life_mask, "out": x}

    def alive_cells(self, x: torch.Tensor):
        """Returns alive mask

        Args:
            x (torch.Tensor): to be masked input

        Returns:
            torch.Tensor: mask. 0 where masking else 1
        """
        return F.max_pool3d(
            x[:, self.num_categories : self.num_categories + 1, :, :, :],
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def light_state(self, shape: torch.Tensor, state: int=0):
        """State dependend perturbation

        Args:
            shape (torch.Tensor): Expected shape for perturbation Tensor
            state (int, optional): State the system is in 0 or 1. Defaults to 0.
        Returns:
            torch.Tensor: Perturbation Tensor
        """

        # light off
        if state == 0:
            x = (
                torch.sin(
                    torch.arange(self.num_hidden_channels, requires_grad=False)
                )
                + 1
            ) * 0.5
            return x.view(-1, 1, 1, 1).expand(self.num_hidden_channels, *shape)
        # light on
        elif state == 1:
            x = (
                torch.cos(
                    torch.arange(self.num_hidden_channels, requires_grad=False)
                )
                + 1
            ) * 0.5
            return x.view(-1, 1, 1, 1).expand(self.num_hidden_channels, *shape)
