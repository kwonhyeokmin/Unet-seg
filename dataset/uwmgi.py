import os.path as osp
import constants
import pandas as pd
from tqdm import tqdm


class UWMGIDataset:
    def __init__(self):
        self.ann_path = osp.join(constants.DATASET_FOLDER, 'train_preprocessed.csv')
        self.cat_name = ['large_bowel', 'small_bowel', 'stomach']
        self.data = self.load_data()

    def load_data(self):
        print(f'Load data that annotation file path is {self.ann_path} ...')
        data = []

        db = pd.read_csv(self.ann_path)
        db = db.dropna()
        for i in tqdm(range(len(db))):
            ann = db.iloc[i]
            data_id = int(str(ann['case']) + str(ann['day']) + str(ann['slice']))
            _prev = [x for x in data if x['id'] == data_id]
            cat = {
                'category_id': self.cat_name.index(ann['class']),
                'segmentation': ann['segmentation']
            }

            if len(_prev) > 0:
                if not pd.isna(cat['segmentation']):
                    _prev[0]['anno'].append(cat)
            else:
                _ann = {
                    'id': data_id,
                    'image_path': ann['image_path'],
                    'height': ann['height'],
                    'width': ann['width'],
                    'case': ann['case'],
                    'day': ann['day'],
                    'slice': ann['slice'],
                    'anno': [],
                }
                if not pd.isna(cat['segmentation']):
                    _ann['anno'].append(cat)
                data.append(_ann)
        print(f'End loading data. The number of data: {len(data)}')
        return data