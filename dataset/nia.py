import os
import os.path as osp
import json
import constants
from glob import glob
import cv2
import pydicom


class NIADataset:
    def __init__(self):
        self.ann_path = [x.replace(os.sep, '/') for x in
                         glob(f'{constants.DATASET_FOLDER}/2.라벨링데이터/**/AXL/*.json', recursive=True)]
        self.cat_name = ['Talus', 'Tibia', 'Fibula', 'MidFoot', 'Calcaneus',
                         '1st Metararsal', '2st Metararsal', '3st Metararsal', '4st Metararsal', '5st Metararsal']
        self.rle = False
        self.data = self.load_data()

    def load_data(self):
        data = []
        data_id = 0

        for ann_path in self.ann_path:
            image_path = ann_path.replace('2.라벨링데이터', '1.원천데이터').replace('.json', '.dcm')
            dcm = pydicom.dcmread(image_path)
            img = dcm.pixel_array
            h, w = img.shape
            _ann = {
                'id': data_id,
                'image_path': image_path,
                'height': h,
                'width': w,
                'anno': []
            }
            with open(ann_path) as fp:
                db = json.load(fp)['ArrayOfannotation_info']
            _seg_str = ''
            for anno in db:
                try:
                    polyline = anno['polyline_list'][0]
                    seg = [[x['Y'], x['X']] for x in polyline['pos_list']]
                    cat = {
                        'category_id': self.cat_name.index(anno['preset_name']),
                        'segmentation': seg
                    }
                    _ann['anno'].append(cat)
                except KeyError as e:
                    print(f'Annotation file has invalid Key: {ann_path}')
                    continue
            data.append(_ann)
            data_id += 1

        print(f'End loading data. The number of data: {len(data)}')
        return data


if __name__ == '__main__':
    dataset = NIADataset()
