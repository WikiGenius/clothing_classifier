import zipfile
import pandas as pd
import os
import pathlib
from tqdm import tqdm
import shutil


def preperare_csv_files(source_dataset, dataset_path):
    if not os.path.isdir(f"{dataset_path}"):
        os.mkdir(f"{dataset_path}")

    data = dict()
    if not os.path.isfile(f'{dataset_path}/list_eval_partition.csv'):
        list_eval_partition_file = f'{source_dataset}/Eval/list_eval_partition.txt'
        list_eval_partition_df = pd.read_csv(
            list_eval_partition_file, skiprows=1,  delim_whitespace=True)
        list_eval_partition_df.to_csv(
            f'{dataset_path}/list_eval_partition.csv', sep=',', index=False)
    data['list_eval_partition'] = pd.read_csv(
        f'{dataset_path}/list_eval_partition.csv')

    if not os.path.isfile(f'{dataset_path}/list_category_cloth.csv'):
        list_category_cloth_file = f'{source_dataset}/Anno_coarse/list_category_cloth.txt'
        list_category_cloth_df = pd.read_csv(
            list_category_cloth_file, skiprows=1,  delim_whitespace=True)
        list_category_cloth_df.to_csv(
            f'{dataset_path}/list_category_cloth.csv', sep=',', index=False)
    data['list_category_cloth'] = pd.read_csv(
        f'{dataset_path}/list_category_cloth.csv')

    if not os.path.isfile(f'{dataset_path}/list_category_img.csv'):
        list_category_img_file = f'{source_dataset}/Anno_coarse/list_category_img.txt'
        list_category_img_df = pd.read_csv(
            list_category_img_file, skiprows=1,  delim_whitespace=True)
        list_category_img_df.to_csv(
            f'{dataset_path}/list_category_img.csv', sep=',', index=False)
    data['list_category_img'] = pd.read_csv(
        f'{dataset_path}/list_category_img.csv')

    return data


def prepare_images(source_dataset_path, dataset_path, generate_images=True, extract=False):
    if generate_images:
        category = dict()

        data = preperare_csv_files(source_dataset_path, dataset_path)
        for split in ['train', 'val', 'test']:
            pathlib.Path(
                f'{dataset_path}/images_v2/{split}').mkdir(parents=True, exist_ok=True)
        df_split = data['list_eval_partition']

        path_imgs = f'{source_dataset_path}/Img/img.zip'
        if extract:
            with zipfile.ZipFile(path_imgs, 'r') as zf:
                for member in tqdm(zf.infolist(), desc='Extracting '):
                    try:
                        zf.extract(member, f"{dataset_path}/imgs_v2")
                    except zipfile.error as e:
                        pass
        for i in tqdm(range(df_split.shape[0]), desc="Copying images "):
            local_image_path = df_split.iloc[i]['image_name']
            split = df_split.iloc[i]['evaluation_status']
            image_path = f'{f"{dataset_path}/imgs_v2"}/{local_image_path}'

            folder_name, image_name = image_path.split('/')[3:]
            category_name = folder_name.split('_')[-1]
            if category_name in category:
                if split in category[category_name]:
                    category[category_name][split] += 1
                else:
                    category[category_name][split] = 1
            else:
                category[category_name] = dict()
                category[category_name][split] = 1
            n = category[category_name][split]
            image_name_updated = f'img_{str(n).zfill(8)}.jpg'
            pathlib.Path(
                f'{dataset_path}/images_v2/{split}/{category_name}').mkdir(parents=True, exist_ok=True)

            new_image_path = f"{dataset_path}/images_v2/{split}/{category_name}/{image_name_updated}"

            shutil.copyfile(image_path, new_image_path)
