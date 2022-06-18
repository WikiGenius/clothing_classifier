# Author: Muhammed El-Yamani

import matplotlib.pyplot as plt
import numpy as np
import os
import re


def find_last_sub_folders_files(folder_name):
    if os.path.isdir(folder_name):
        check_folders = sorted(os.listdir(folder_name))
        if len(check_folders) > 0:
            nums = list()
            for folder in check_folders:
                match = re.search(r'[0-9]+', folder)
                if match:
                    nums.append(int(match.group()))
            nums.sort()
            if len(nums) > 0:
                last_num = nums[-1] + 1
                return last_num
    last_num = 1
    return last_num


def find_last_version_model(dir_checkpoint: str = None, type_load_model: str = 'load_best'):
    '''
        type_load_model: load_best | load_interrupted | load_last | create_new
    '''
    if dir_checkpoint is None:
        last_num_check_folders = find_last_sub_folders_files(
            './checkpoints') - 1
        dir_checkpoint = f'./checkpoints/v{last_num_check_folders}'
    if type_load_model == 'load_best':
        load_model = f'{dir_checkpoint}/best_checkpoint.pth'
    elif type_load_model == 'load_interrupted':
        load_model = f'{dir_checkpoint}/INTERRUPTED.pth'
    elif type_load_model == 'load_last':
        last_num_epoch_file = find_last_sub_folders_files(
            f'{dir_checkpoint}') - 1
        load_model = f'{dir_checkpoint}/checkpoint_epoch{last_num_epoch_file}.pth'
    elif type_load_model == 'create_new':
        load_model = False
        return load_model, dir_checkpoint
    else:
        raise
    if load_model:
        if not os.path.isfile(load_model):
            load_model = False
    return load_model, dir_checkpoint


def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_title(title)
    return ax
