r""" FSS-1000 few-shot semantic segmentation dataset """
import os
import glob

import random

from dataset.transform import crop, hflip, normalize
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np


class DatasetFSS(Dataset):
    def __init__(self, datapath, split, shot, size, episode):
        self.split = split
        self.benchmark = 'fss'
        self.shot = shot

        self.base_path = os.path.join(datapath, 'fewshot_data')

        # Given predefined test split, load randomly generated training/val splits:
        # (reference regarding trn/val/test splits: https://github.com/HKUSTCV/FSS-1000/issues/7))
        with open('./dataset/fss_splits/%s.txt' % split, 'r') as f:
            self.categories = f.read().split('\n')[:-1]
        self.categories = sorted(self.categories)

        self.crop_size = size
        self.episode = episode

    def __len__(self):
        return self.episode
    
    def __getitem__(self, unused_idx):
        # the sampling strategy is based on the description in OSLSM paper
        # Randomly select a category
        selected_category = random.choice(self.categories)
        cat_folder = os.path.join(self.base_path, selected_category)
        assert os.path.exists(cat_folder) and os.path.isdir(cat_folder)
        
        possible_img_fn_list = [fn for fn in os.listdir(cat_folder) if fn.endswith('.jpg')]
        assert len(possible_img_fn_list) == 10
        
        img_q_fn = random.choice(possible_img_fn_list)
        mask_q_fn = img_q_fn.replace('.jpg', '.png')
        
        img_q = Image.open(os.path.join(cat_folder, img_q_fn)).convert('RGB')
        mask_q = self.read_mask(os.path.join(cat_folder, mask_q_fn))
        
        fn_s_list, img_s_list, mask_s_list = [], [], []
        while True:
            img_s_fn = random.choice(list(set(possible_img_fn_list) - {img_q_fn} - set(fn_s_list)))
            mask_s_fn = img_s_fn.replace('.jpg', '.png')
            
            img_s = Image.open(os.path.join(cat_folder, img_s_fn)).convert('RGB')
            mask_s = self.read_mask(os.path.join(cat_folder, mask_s_fn))
            
            fn_s_list.append(img_s_fn)
            img_s_list.append(img_s)
            mask_s_list.append(mask_s)
            if len(fn_s_list) == self.shot:
                break

        if self.split == 'trn':
            img_q, mask_q = crop(img_q, mask_q, self.crop_size)
            img_q, mask_q = hflip(img_q, mask_q)
            for k in range(self.shot):
                img_s_list[k], mask_s_list[k] = crop(img_s_list[k], mask_s_list[k], self.crop_size)
                img_s_list[k], mask_s_list[k] = hflip(img_s_list[k], mask_s_list[k])

        img_q, mask_q = normalize(img_q, mask_q)
        for k in range(self.shot):
            img_s_list[k], mask_s_list[k] = normalize(img_s_list[k], mask_s_list[k])
        
        cls_idx = self.category_name_to_idx(selected_category)

        return img_s_list, mask_s_list, img_q, mask_q, cls_idx, fn_s_list, img_q_fn

    def read_mask(self, img_name):
        mask = np.array(Image.open(img_name).convert('L'))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        mask = Image.fromarray(mask)
        return mask
    
    def category_name_to_idx(self, category_name):
        relative_offset = self.categories.index(category_name)
        if self.split == 'trn':
            base = 0
        elif self.split == 'val':
            base = 520
        elif self.split == 'test':
            base = 760
        else:
            raise ValueError('Wrong split!')
        
        return base + relative_offset
