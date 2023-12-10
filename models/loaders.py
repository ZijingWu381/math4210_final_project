import os
from typing import Iterable, Optional, Sequence, Union

from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t
if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm

import torch
from torch.utils.data import DataLoader



def custom_collate_fn(batch: Sequence) -> Sequence:
    """
    Args:
        batch: a sequence of (x_trial, u_trial) pairs in a trial
               x_trial has shape (trial_length, dim_x) and
               u_trial has shape (trial_length, dim_u)
    """
    data = list(zip(*batch))
    if len(data) == 2:
        x, u = data[0], data[1]
        x, u = torch.cat(x), torch.cat(u)
        return x, u
    elif len(data) == 4:
        x, u, trial_ids, time_stamps = data[0], data[1], data[2], data[3]
        x, u, trial_ids, time_stamps = torch.cat(x), torch.cat(u), torch.cat(trial_ids), torch.cat(time_stamps)
        return x, u, trial_ids, time_stamps


    # return torch.cat(x), torch.cat(u)

class DataLoaderWithCollate(DataLoader):
    def __init__(self, dataset, batch_size, shuffle):
        super().__init__(dataset, batch_size, shuffle, collate_fn=custom_collate_fn)
