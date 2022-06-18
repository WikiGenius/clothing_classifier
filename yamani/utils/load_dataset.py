from pathlib import Path
from PIL import Image
import torch
import numpy as np
import glob
from torchvision import transforms
from collections import Counter
import matplotlib.pyplot as plt
import os


def image_transformations(augment: bool, new_size_hw: tuple):
    if augment:
        image_transforms = transforms.Compose([
            transforms.RandomOrder([
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(new_size_hw, scale=(0.8, 1), ratio=(0.8, 1.2), interpolation=transforms.InterpolationMode.BICUBIC)]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    else:
        image_transforms = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(new_size_hw),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    return image_transforms


def label_transforms(label_category, class_names):
    label = np.where(class_names == label_category)[0][0]
    return label.astype(np.int64)


def transform_data_entry(image_path, class_names, augment,  new_size_hw: tuple):
    image_transforms = image_transformations(augment, new_size_hw)
    label_category = image_path.split('/')[-2]
    image = Image.open(image_path)
    
    # image = np.asarray(image, dtype=np.float64)
    # image = Image.fromarray(image)

    image = image_transforms(image).contiguous()
    label = label_transforms(label_category, class_names)

    return image, label


def get_classes(image_file_list):

    list_categories = [path.split('/')[-2] for path in sorted(image_file_list)]
    freq_classes_train = Counter(list_categories).most_common()
    class_names = np.unique(sorted(list_categories))
    return class_names, freq_classes_train


def random_keep_images_list(root, origin_image_file_list, keep_data_factor):
    class_names, freq_classes = get_classes(origin_image_file_list)
    image_file_list = list()
    new_size = int(keep_data_factor * len(origin_image_file_list))
    size_accum = 0
    for index, (class_name, value_counts) in enumerate(freq_classes[::-1]):
        thresh_keep = int((new_size - size_accum)/(len(class_names) - index))

        path_cont = f'{root}/{class_name}/*.jpg'
        class_name_list = np.array(glob.glob(path_cont))
        class_name_list = list(class_name_list[np.random.permutation(len(class_name_list))[
            :thresh_keep]].copy())
        image_file_list.extend(class_name_list)
        size_accum += len(class_name_list)

    return image_file_list


class ClothesDataset(torch.utils.data.Dataset):
    def __init__(self, root, aug,  new_size_hw: tuple, keep_data_factor):
        images_path = f"{root}/**/*.jpg"
        origin_image_file_list = glob.glob(images_path)
        assert len(origin_image_file_list) > 0
        image_file_list = random_keep_images_list(
            root, origin_image_file_list, keep_data_factor)

        self.split = root.split('/')[-1]
        self.keep_data_factor = keep_data_factor
        self.aug = aug
        self.image_file_list = image_file_list
        self.new_size_hw = new_size_hw

        self.class_names, self.freq_classes = get_classes(self.image_file_list)

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, idx):
        image_path = self.image_file_list[idx]
        image, label = transform_data_entry(
            image_path, self.class_names, self.aug,  new_size_hw=self.new_size_hw)
        return image, label

    def plot_distribution(self, version='v1'):
        Path(f'results').mkdir(parents=True, exist_ok=True)
        if not os.path.isfile(f'results/freq_classes_{self.split}_{version}.png'):
            x, y = zip(*self.freq_classes)
            plt.figure(figsize=[12, 12])
            plt.bar(x, y)
            plt.xticks(rotation=90)
            plt.savefig(f'results/freq_classes_{self.split}_{version}.png')
            plt.show()

        else:
            img = Image.open(
                f'results/freq_classes_{self.split}_{version}.png')
            plt.imshow(img)


def get_dataset_dataloader(dataset_path: str, image_size: tuple, split: str, keep_data_factor: float = 0.1, seed: int = 0, aug_train: bool = True, shuffle: bool = True, loader_args=dict(batch_size=32, num_workers=2, pin_memory=True)):
    '''
    split: train | val | test   
    keep_data_factor: range(0, 1) 
    To test on small sample of data
    new_data_size = keep_data_factor * data_size
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    data_dir = f'{dataset_path}/images_v2/{split}'
    if split == 'train' and aug_train == True:
        aug = True
    else:
        aug = False
    dataset = ClothesDataset(data_dir, aug,  image_size, keep_data_factor)
    print(f'{split} dataset: {len(dataset)} images')

    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=shuffle, **loader_args)

    return dataset, dataloader
