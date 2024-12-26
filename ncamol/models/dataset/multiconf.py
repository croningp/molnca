import torch
from torch.utils.data import Dataset


class MultiConfDataset(Dataset):
    def __init__(
        self, conf_cis, conf_trans, pretrain=True
    ) -> None:
        super().__init__()

        if pretrain:
            self.train_data = {
                0: {
                    "input": torch.cat([conf_trans], dim=0).squeeze(0),
                    "target": torch.cat([conf_cis], dim=0).squeeze(0),
                    "state": 0
                },
                1: {
                    "input": torch.cat([conf_cis], dim=0).squeeze(0),
                    "target": torch.cat([conf_trans], dim=0).squeeze(0),
                    "state": 1
                },
            }
        else:
            self.train_data = {
                0: {
                    "input": torch.cat([conf_trans, conf_cis], dim=0),
                    "target": torch.cat([conf_cis, conf_cis], dim=0),
                    "state": 0
                },
                1: {
                    "input": torch.cat([conf_cis, conf_trans], dim=0),
                    "target": torch.cat([conf_trans, conf_trans], dim=0),
                    "state": 1
                },
            }

    def __len__(self) -> int:
        return len(self.train_data)
    
    def __getitem__(self, idx):
        return self.train_data[idx]