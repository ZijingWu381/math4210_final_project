import os

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import numpy as np
import scipy.io as sio


class DataSet(object):
    '''
    Base class for representing a dataset. Meant to be subclassed
    '''
    def __init__(self, ds_name, data_path, **kwargs):
        """
        Args:
            ds_name (str) : string identifier for the dataset
            data_path (str) : path to the dataset 
        """
        # TODO: add required args to as we see fit
        required_args = []
        assert set(required_args).issubset(set(kwargs.keys())), "Missing required args, only saw %s" % kwargs.keys()
        self.ds_name = ds_name
        self.data_path = data_path
        self.__dict__.update(kwargs)


class RatHippo(DataSet):
    """
    Rat hippocampus place cell dataset (TODO: source)
    """

    def __init__(self, data_path='../data/achilles_data/Achilles_data.mat', split='train', 
                    model_driven_mode=None, u_label=None, num_bin=None,
                    **kwargs):
        """
        """
        super(RatHippo, self).__init__('rat_hippo', data_path, **kwargs)
        self.model_driven_mode = model_driven_mode
        self.u_label = u_label
        self.num_bin = num_bin

        loaded_data = sio.loadmat(data_path)

        ## load trial information
        self.idx_split = loaded_data["trial"][0]
        ## load spike data
        self.spike_by_neuron_use = loaded_data["spikes"]
        ## load locations
        self.locations_vec = loaded_data["loc"][0]

        self.dim_u = None
        u_all = self.get_u_all()
        x_all = self.get_x_all()

        assert split is None or split in ['train', 'val', 'test', 'trainval'], 'got {}'.format(split)
        self.split = split

        # self.x, self.u = self._split_train_test(x_all, u_all, split=split)
        self.data = self._split_train_test(x_all, u_all, split=split)

    def _split_train_test(self, x_all, u_all, split, train_split_idx=68, valid_split_idx=76):
        if split is None:
            x, u = x_all, u_all

        trial_ls = np.arange(len(u_all))

        if split == 'test':
            u_test = u_all[trial_ls[valid_split_idx:]]
            x_test = x_all[trial_ls[valid_split_idx:]]
            x, u = x_test, u_test
        else:
            if split == 'trainval':
                x_trainval = x_all[trial_ls[:valid_split_idx]]
                u_trainval = u_all[trial_ls[:valid_split_idx]]
                x, u = x_trainval, u_trainval
            else:
                if split == 'train':
                    u_train = u_all[trial_ls[:train_split_idx]]
                    x_train = x_all[trial_ls[:train_split_idx]]
                    x, u = x_train, u_train
                elif split == 'val':
                    u_valid = u_all[trial_ls[train_split_idx:valid_split_idx]]
                    x_valid = x_all[trial_ls[train_split_idx:valid_split_idx]]
                    x, u = x_valid, u_valid
        
        # Cast to torch tensors
        x = [torch.tensor(d, dtype=torch.float32) for d in x]
        u = [torch.tensor(d, dtype=torch.float32) for d in u]

        data = list(self.zip_data(x, u))

        return data
    
    def add_trial_id_to_data(self):
        """
        Adds trial id to the data.
        """
        for ii in range(len(self.data)):
            trial_length = len(self.data[ii][0])
            self.data[ii] = list(self.data[ii]) + [torch.ones(trial_length, dtype=torch.int) * ii]
            self.data[ii] = tuple(self.data[ii])
        
    def add_time_stamp_to_data(self):
        """
        Adds time stamp to the data.
        """
        for ii in range(len(self.data)):
            trial_length = len(self.data[ii][0])
            self.data[ii] = list(self.data[ii]) + [torch.arange(trial_length, dtype=torch.int)]
            self.data[ii] = tuple(self.data[ii])
        
    def add_trial_id_and_time_stamp_to_data(self):
        """
        Adds trial id and time stamp to the data.
        """
        for ii in range(len(self.data)):
            trial_length = len(self.data[ii][0])
            self.data[ii] = list(self.data[ii])\
                            + [torch.ones(trial_length, dtype=torch.int) * ii]\
                            + [torch.arange(trial_length, dtype=torch.int)]
            self.data[ii] = tuple(self.data[ii])

    def zip_data(self, x, u):
        """
        Zips the provided data for neural activity (x) and inputs (u) across trials and timepoints.
        
        Args:
            x (list): (num_trial, num_timepoint, num_neuron)
            u (list): (num_trial, num_timepoint, num_bin)
        """
        # return list(zip(x, u))
        return [(x[ii], u[ii]) for ii in range(len(x))]

    def __getitem__(self, index):
        # return self.x[index], self.u[index]
        return self.data[index]

    def __len__(self):
        return len(self.data)
        # return len(self.x)
    
    @property
    def trial_lengths(self):
        return [len(d[0]) for d in self.data]


    def get_x_all(self):
        idx_split = self.idx_split
        spike_by_neuron_use = self.spike_by_neuron_use

        x_all = np.array(np.array_split(spike_by_neuron_use, idx_split[1:-1], axis=0), dtype=object)
        return x_all

    def get_u_all(self, u_label=None):
        if u_label is None:
            u_label = self.u_label

        idx_split = self.idx_split
        spike_by_neuron_use = self.spike_by_neuron_use
        locations_vec = self.locations_vec

        u_all = np.array(
            np.array_split(
                np.hstack((locations_vec.reshape(-1, 1), np.zeros((locations_vec.shape[0], 2)))), idx_split[1:-1], axis=0
            ), dtype=object
        )
        for ii in range(len(u_all)):
            u_all[ii][:, int(ii % 2) + 1] = 1

        if self.model_driven_mode is None:
            u_all = np.array(np.array_split(np.zeros((locations_vec.shape[0], 1)), idx_split[1:-1], axis=0), dtype=object)
        
        # Get the labels in u specified by u_label
        if u_label is not None:
            u_all = self.get_u(u_all, label=u_label, num_bin=self.num_bin)

        if self.dim_u is None:
            self.dim_u = u_all[0].shape[-1]

        return u_all

    def get_u(self, u_loc_dir, label=None, num_bin=None):
        """ Rat hippocampus data u getter"""
        assert label in ["loc_dir", "loc_time", "dir_time", "time", "location", "direction", "loc_dir_time"]

        u = []

        for trial, u_trial in enumerate(u_loc_dir):
            trial_length = len(u_trial)
            order = np.arange(0, trial_length).reshape(-1, 1)
            order = order / np.max(order)  # normalize time
            assert order[-1] == 1, order[-1]
            assert order[0] == 0, order[0]

            loc = u_trial[:, 0].reshape(-1, 1)
            dir = u_trial[:, 1:]

            if num_bin is not None:
                order = np.digitize(order, np.linspace(0, 1, num_bin)).astype(int).reshape(-1, 1)
                loc = np.digitize(loc, np.linspace(0, 1.6, num_bin)).astype(int).reshape(-1, 1)

            if label == "loc_dir":
                # u.append(u_trial)
                if num_bin is not None: 
                    # create unique index for each [location, direction] pair
                    min_u = np.ones(loc.shape)   # this is hard coded, should be changed if num_bin is changed 
                    ind = loc + (dir[:, 0] * num_bin).reshape(-1, 1) - min_u
                    u.append(ind)
                else:
                    u.append(np.concatenate([loc, dir], axis=1))
            elif label == "loc_time":
                u.append(np.concatenate([loc, order], axis=1))
            elif label == "dir_time":
                u.append(np.concatenate([dir, order], axis=1))
            elif label == "time":
                u.append(order)
            elif label == "location":
                u.append(loc)
            elif label == "direction":
                u.append(dir)
            elif label == "loc_dir_time":
                u.append(np.concatenate([loc, dir, order], axis=1))

        return np.array(u, dtype=object)


    
class SwissRoll(DataSet):
    """
    Swiss roll dataset. The data is generated with sklearn.datasets.make_swiss_roll
    """

    def __init__(self, data_path=None, split='train', 
                    model_driven_mode=None, u_label=None, num_bin=None,
                    **kwargs):
        """
        """
        super(SwissRoll, self).__init__('swiss_roll', data_path, **kwargs)
        self.model_driven_mode = model_driven_mode
        self.u_label = u_label
        self.num_bin = num_bin

        from sklearn.datasets import make_swiss_roll
        # Generate Swiss Roll dataset
        # The data should be randomized already
        n_samples = 10000
        noise = 0.5
        X, t = make_swiss_roll(n_samples=n_samples, noise=noise)

        self.dim_u = None

        self.X = X  
        self.t = t

        x_all = self.get_x_all()
        u_all = self.get_u_all()


        assert split is None or split in ['train', 'val', 'test', 'trainval'], 'got {}'.format(split)
        self.split = split

        # self.x, self.u = self._split_train_test(x_all, u_all, split=split)
        self.data = self._split_train_test(x_all, u_all, split=split)

    def get_x_all(self):
        x_all = self.X.reshape(100, 100, 3)
        # x_all = self.X.reshape(100, 1000, 3)
        return x_all
    
    def get_u_all(self, u_label=None):
        u_all = self.t.reshape(100, 100, 1)
        # u_all = self.t.reshape(100, 1000, 1)
        x = self.get_x_all()
        y = x[:, :, 1].reshape(100, 100, 1)
        # y = x[:, :, 1].reshape(100, 1000, 1)
        u_all = np.concatenate([u_all, y], axis=-1)
        if self.dim_u is None:
            self.dim_u = u_all[0].shape[-1]
        return u_all

    def _split_train_test(self, x_all, u_all, split, train_split_idx=60, valid_split_idx=80):
        if split is None:
            x, u = x_all, u_all

        trial_ls = np.arange(len(u_all))

        if split == 'test':
            u_test = u_all[trial_ls[valid_split_idx:]]
            x_test = x_all[trial_ls[valid_split_idx:]]
            x, u = x_test, u_test
        else:
            if split == 'trainval':
                x_trainval = x_all[trial_ls[:valid_split_idx]]
                u_trainval = u_all[trial_ls[:valid_split_idx]]
                x, u = x_trainval, u_trainval
            else:
                if split == 'train':
                    u_train = u_all[trial_ls[:train_split_idx]]
                    x_train = x_all[trial_ls[:train_split_idx]]
                    x, u = x_train, u_train
                elif split == 'val':
                    u_valid = u_all[trial_ls[train_split_idx:valid_split_idx]]
                    x_valid = x_all[trial_ls[train_split_idx:valid_split_idx]]
                    x, u = x_valid, u_valid
        
        # Cast to torch tensors
        x = [torch.tensor(d, dtype=torch.float32) for d in x]
        u = [torch.tensor(d, dtype=torch.float32) for d in u]

        data = list(self.zip_data(x, u))

        return data
    
    def add_trial_id_to_data(self):
        """
        Adds trial id to the data.
        """
        for ii in range(len(self.data)):
            trial_length = len(self.data[ii][0])
            self.data[ii] = list(self.data[ii]) + [torch.ones(trial_length, dtype=torch.int) * ii]
            self.data[ii] = tuple(self.data[ii])
        
    def add_time_stamp_to_data(self):
        """
        Adds time stamp to the data.
        """
        for ii in range(len(self.data)):
            trial_length = len(self.data[ii][0])
            self.data[ii] = list(self.data[ii]) + [torch.arange(trial_length, dtype=torch.int)]
            self.data[ii] = tuple(self.data[ii])
        
    def add_trial_id_and_time_stamp_to_data(self):
        """
        Adds trial id and time stamp to the data.
        """
        for ii in range(len(self.data)):
            trial_length = len(self.data[ii][0])
            self.data[ii] = list(self.data[ii])\
                            + [torch.ones(trial_length, dtype=torch.int) * ii]\
                            + [torch.arange(trial_length, dtype=torch.int)]
            self.data[ii] = tuple(self.data[ii])

    def zip_data(self, x, u):
        """
        Zips the provided data for neural activity (x) and inputs (u) across trials and timepoints.
        
        Args:
            x (list): (num_trial, num_timepoint, num_neuron)
            u (list): (num_trial, num_timepoint, num_bin)
        """
        # return list(zip(x, u))
        return [(x[ii], u[ii]) for ii in range(len(x))]

    def __getitem__(self, index):
        # return self.x[index], self.u[index]
        return self.data[index]

    def __len__(self):
        return len(self.data)
        # return len(self.x)
    
    @property
    def trial_lengths(self):
        return [len(d[0]) for d in self.data]
