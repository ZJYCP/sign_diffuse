import pickle
import random
import numpy as np
from torch.utils import data
import torch
from os.path import join as pjoin

class SignDataset(data.Dataset):
    def __init__(self, data_dir):
        super().__init__()

        self.dataset = pickle.load(open(data_dir, 'rb'))
        self.max_motion_length = 300

    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        data = self.dataset[index]

        motion, m_length, text, gloss = data['pose'], data['dur'], data['text'], data['gloss']
        motion = np.array(motion)

        max_motion_length = self.max_motion_length
        if m_length >= self.max_motion_length:
            # idx = random.randint(0, len(motion) - max_motion_length)
            # motion = motion[idx: idx + max_motion_length]
            motion = motion[:max_motion_length]
        else:
            padding_len = max_motion_length - m_length
            D = motion.shape[1]
            padding_zeros = np.zeros((padding_len, D))
            motion = np.concatenate((motion, padding_zeros), axis=0)                         
        return text, gloss, torch.from_numpy(motion).float(), m_length