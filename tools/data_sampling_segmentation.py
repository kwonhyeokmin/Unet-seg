from glob import glob
import json
import argparse
import pandas as pd
from tqdm import tqdm
import math
from collections import deque
import pydicom
import shutil
import os


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', required=True, help='Image type for sampling (AXL, SAG, COR)', choices=['AXL', 'SAG', 'COR'])
    args = parser.parse_args()
    return args


def print_statistics(df, std_cat):
    # *******************
    # Print Statistics
    # *******************
    print('Dataframe Sample')
    print('--------------------')
    print(df.head(5))
    print()

    # statistic of disease
    for cat in std_cat:
        print(f'Statistics of {cat}')
        print('--------------------')
        print(df.groupby(f'{cat}')[f'{cat}'].count())
        print()


def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def copy_files_with_parents(file_path, base_dir, desc):
    try:
        target_d = f'{desc}{base_dir}'
        make_folder(target_d)
        shutil.copy2(file_path, target_d)
    except Exception as e:
        print(e)
        return False
    return True


def divided_algorithm(df, queue, N):
    _df = df.copy()
    # Calculate the number of images by case_id
    std_col = queue.popleft()
    C = set(_df[std_col])

    divided = pd.DataFrame(columns=_df.columns)
    N_count = _df.groupby(std_col)[std_col].count()
    while len(C) > 0 and N > 0:
        N_std = N_count.min()
        C_std = N_count.idxmin()

        if N_std < N / len(C):
            N = N - N_std
            C.remove(C_std)
            divided = pd.concat([divided, pd.DataFrame(_df[_df[std_col]==C_std])], ignore_index=True)
            _df.drop(_df[_df[std_col]==C_std].index, inplace=True)
            N_count = N_count.drop(labels=C_std)
        else:
            # recursive
            if len(queue) > 0:
                result, N = divided_algorithm(_df, queue, N)
                divided = pd.concat([divided, result], ignore_index=True)
            else:
                # choice by random
                for c in C:
                    sample_c = _df[_df[std_col]==c].sample(n=math.ceil(N / len(C)), random_state=1)
                    divided = pd.concat([divided, sample_c], ignore_index=True)
                return divided, 0
    return divided, N


if __name__ == '__main__':
    # args
    args = make_args()

    # Directory setting
    DATA_ROOT = '/workspace/data/1.Datasets'
    DES_ROOT = '/workspace/data/ori/1.Datasets'
    tag = args.data_type
    json_paths = glob(f'{DATA_ROOT}/2.라벨링데이터/**/{tag}/*.json', recursive=True)
    data_ratio = 0.5

    # Setting categories name
    cat_nm = ['1st Metatarsal', '2nd Metatarsal', '3rd Metatarsal', '4th Metatarsal', '5th Metatarsal',
              'Calcaneus', 'Fibula', 'MidFoot', 'Talus', 'Tibia']

    print('Load json files ...')
    infos = []
    for path in tqdm(json_paths, desc='load json'):
        with open(path, 'r') as fp:
            db = json.load(fp)
        if len(db['ArrayOfannotation_info']) < 1:
            print(path)
            continue

        # clinical info
        clinical_info = db['ClinicalInfo']
        case_id = clinical_info['Case_ID']
        age = clinical_info['Age']
        sex = clinical_info['Sex']
        for ann in db['ArrayOfannotation_info']:
            if ann['object_name'] != 'LabelPolyLine':
                continue
            _cat_nm = ann['preset_name']
            if _cat_nm not in cat_nm:
                continue
            _info = {
                'case_id': case_id,
                'path': path,
                'sex': sex,
                'Age': age,
                'generation': age // 10,
                'cat_nm': _cat_nm
            }
            infos.append(_info)

    df = pd.DataFrame.from_records(infos)
    print_statistics(df, std_cat=['cat_nm', 'sex', 'generation'])

    # data sampling
    print('Divided data ...', 'red')
    N_p = len(df.index)
    N = math.ceil(N_p * data_ratio)
    df_divided, _ = divided_algorithm(df, deque(['case_id', 'cat_nm', 'generation', 'sex']), N)
    print_statistics(df_divided, std_cat=['cat_nm', 'sex', 'generation'])

    # Copy files
    print('Copy files that sampled', 'red')
    json_info = []
    case_ids = df_divided['case_id'].unique()
    for case_id in tqdm(case_ids):
        _info = {'case_id': case_id}
        # label json
        seg_label_paths = [x for x in
                           glob(f'{DATA_ROOT}/2.라벨링데이터/**/{case_id}/**/{tag}/*.json', recursive=True)]
        for lpath in seg_label_paths:
            # check image path is valid
            img_path = lpath.replace('2.라벨링데이터', '1.원천데이터').replace('.json', '.dcm')
            dcm = pydicom.dcmread(img_path)
            dcm_img = dcm.pixel_array
            if dcm_img is None:
                print(f'File not exist path is {img_path}')
                continue
            # copy images and label files to destination directory
            copy_files_with_parents(img_path, os.path.dirname(img_path.replace(DATA_ROOT, '')), DES_ROOT)
            copy_files_with_parents(lpath, os.path.dirname(lpath.replace(DATA_ROOT, '')), DES_ROOT)
