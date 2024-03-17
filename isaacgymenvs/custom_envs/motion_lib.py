import numpy as np
import os
import yaml
import zarr
import torch
import random
import copy

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def pair_data(data_dict, episode_ends, num_amp_obs_steps, num_amp_obs_per_step, device):
    """
    Create sets of data [ [s1,s2,.. sn], [s2,s3,.. s+1] ...]
    n = num_amp_obs_steps
    """
    for key, data in data_dict.items():
        # paired_size = data.shape[0] - len(episode_ends)*(num_amp_obs_steps - 1)
        # paired_data = torch.zeros((paired_size, num_amp_obs_steps*num_amp_obs_per_step), device=device, dtype=torch.float)

        # new ends after having paired data
        new_ends = copy.deepcopy(episode_ends) # - (num_amp_obs_per_step - 1)

        # preprocess episode ends list
        episode_ends = episode_ends.tolist()
        if data.shape[0] in episode_ends:
            episode_ends.remove(data.shape[0])
            episode_ends.append(data.shape[0]-1)

        # list of indices to delete after pairing up data
        del_list = []
        for end in episode_ends:
            for i in range(num_amp_obs_steps - 1):
                if i != 0:
                    del_list.append((end - i))
        del_list = del_list + episode_ends

        # make copies of original data and shift each copy progressively by one index
        shifted_copies = []
        for i in range(num_amp_obs_steps - 1):
            shifted_array = np.copy(data)
            shifted_array[:-1-i,:] = shifted_array[1+i:,:]
            # shifted_array[-1-i:,:] = 0.0
            # shifted_array = np.delete(shifted_array, [-1-i:])
            shifted_copies.append(shifted_array)


        # delete unnecessary indices from all arrays
        data = np.delete(data, del_list, axis=0)
        for idx, arr in enumerate(shifted_copies):
            arr = np.delete(arr, del_list, axis=0)
            shifted_copies[idx] = arr

        # concatenate all arrays
        shifted_copies.insert(0, data)
        paired_data = np.concatenate(shifted_copies, axis=1)

        paired_episodes = np.split(paired_data, new_ends.tolist())
        episodes = [] 
        for episode in paired_episodes:
            if episode.shape[0] > 0:
                episodes.append(episode)

        return paired_data, episodes


class MotionDataset():
    def __init__(self, motion_file, num_amp_obs_steps, num_amp_obs_per_step, episodic=True, device=None, normalize=False):
        if device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.episodic_sampling = episodic
        self.normalize = normalize
        self.num_amp_obs_steps = num_amp_obs_steps
        self.num_amp_obs_per_step = num_amp_obs_per_step
        self._load_motions(motion_file)
        
        # Episode to load data from
        self.episode = None
        self.offset = 0

    def _load_motions(self, motion_file):
        dataset_root = zarr.open(motion_file, 'r')
        # All demonstration episodes are concatinated in the first dimension N
        train_data = {
            # (N, action_dim)
            # 'action': dataset_root['data']['action'][:],
            # (N, obs_dim)
            'obs': dataset_root['data']['state'][:]
        }
        # Marks one-past the last index for each episode
        episode_ends = dataset_root['meta']['episode_ends'][:]

        if self.normalize:
            # compute statistics and normalized data to [-1,1]
            stats = dict()
            processed_train_data = dict()
            for key, data in train_data.items():
                stats[key] = get_data_stats(data)
                processed_train_data[key] = normalize_data(data, stats[key])

            self.stats = stats
            self.processed_train_data = processed_train_data

        else:
            processed_train_data = train_data
                
        paired_processed_data, paired_processed_episodes = pair_data(processed_train_data, episode_ends, self.num_amp_obs_steps, self.num_amp_obs_per_step, self.device)


        self.paired_processed_data = paired_processed_data
        self.paired_processed_episodes = paired_processed_episodes
        self.num_episodes = len(paired_processed_episodes)

    def __len__(self):
        # all possible segments of the dataset
        return len(self.paired_processed_data)

    def __getitem__(self, idx):
        """ Sample either from paired processed data (not separated by episodes) or from paired processed episodes
        """

        if self.episodic_sampling == True:
            assert self.episode != None, "Please select an episode in the dataloader sample method"
            # try:
            sample = self.paired_processed_episodes[self.episode][idx + self.offset]
            # except IndexError as e:
            #     print(self.paired_processed_episodes[self.episode])
            #     raise e
        else:
            sample = self.paired_processed_data[idx]


        return sample

class MotionLib():
    def __init__(self, motion_file, num_amp_obs_steps, num_amp_obs_per_step, episodic=True, device=None, normalize=False):
        # By default the dataset is normalised. If not needed, it is unnormalized here. 
        # NOTE: AMP also normalizes data internally. It is hence advisable to set normalization to false while sampling and let AMP handle it internally
        self.normalize = normalize

        self.dataset = MotionDataset(motion_file, num_amp_obs_steps, num_amp_obs_per_step, episodic=episodic, device=device, normalize=normalize)
        self.dataloader = None

    def sample_motions(self, num_samples):
        """Sample a trajectory of length num_samples from the demonstration dataset.

        This only samples from episodes that have trajectories longer than num_samples. Naturally, samples in the batch are not shuffled 
        """
        if self.dataloader == None:
            self._setup_trajectory_dataloader(num_samples)

        ep_found = False
        while not ep_found:
            random_episode_idx = random.randrange(self.dataset.num_episodes)
            if len(self.dataset.paired_processed_episodes[random_episode_idx]) > num_samples:
                ep_found = True
        self.dataset.episode = random_episode_idx
        self.dataset.offset = random.randrange((len(self.dataset.paired_processed_episodes[random_episode_idx])) - num_samples)

        batch = next(iter(self.dataloader))

        return batch

    def get_episodes(self):
        return self.dataset.paired_processed_episodes

    def _setup_trajectory_dataloader(self, batch_size):
        """Set up a dataloader on the motion dataset to sample unshuffled motion trajectories
        """
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=1,
            shuffle=False,
            # accelerate cpu-gpu transfer
            pin_memory=True,
        )

    def get_traj_agnostic_dataloader(self, batch_size, shuffle=False):
        """Returns a dataloader that can be used to sample a batch of observation pairs regardless of the episode endings. 

        Primarily used to train trajectory-agnostic methods like diffusion.
        """
        traj_agnostic_dataloader = torch.utils.data.DataLoader(
                    self.dataset,
                    batch_size=batch_size,
                    num_workers=1,
                    shuffle=shuffle,
                    # accelerate cpu-gpu transfer
                    pin_memory=True,
                    )

        return traj_agnostic_dataloader