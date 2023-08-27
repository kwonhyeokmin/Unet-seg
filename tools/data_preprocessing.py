import pandas as pd
import argparse
from glob import glob
from tqdm import tqdm
import os.path as osp


def get_metadata(row):
    data = row['id'].split('_')
    case = int(data[0].replace('case',''))
    day = int(data[1].replace('day',''))
    slice_ = int(data[-1])
    row['case'] = case
    row['day'] = day
    row['slice'] = slice_
    return row


def path2info(row):
    path = row['image_path']
    data = path.split('/')
    slice_ = int(data[-1].split('_')[1])
    case = int(data[-3].split('_')[0].replace('case',''))
    day = int(data[-3].split('_')[1].replace('day',''))
    width = int(data[-1].split('_')[2])
    height = int(data[-1].split('_')[3])
    row['height'] = height
    row['width'] = width
    row['case'] = case
    row['day'] = day
    row['slice'] = slice_
    return row


def preprocessing(df, img_dir):
    print('Start preprocessing')
    print('Preprocessing related with meta data ...')
    tqdm.pandas()

    df = df.progress_apply(get_metadata, axis=1)
    print('... End meta data preprocessing')

    print('Preprocessing related with images ...')
    paths = glob(osp.join(f'{img_dir}/**/*.png'), recursive=True)
    path_df = pd.DataFrame(paths, columns=['image_path'])
    path_df = path_df.progress_apply(path2info, axis=1)
    df = df.merge(path_df, on=['case', 'day', 'slice'])
    print('... End meta images preprocessing')
    print('End preprocessing!')
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', required=True, help='Path of image directories')
    parser.add_argument('--anno_path', required=True, help='Path of annotation files')
    parser.add_argument('--output_path', required=True, help='Path of output file')
    args = parser.parse_args()

    df = pd.read_csv(args.anno_path)
    preprocessed_df = preprocessing(df, args.img_path)

    preprocessed_df.to_csv(args.output_path, sep=',', na_rep='NaN')
