import os
import os.path as osp
import json
import constants
from glob import glob
import cv2
import pydicom


class NIADataset:
    def __init__(self, data_split):
        self.ann_path = []
        self.ann_path += [x.replace(os.sep, '/') for x in
                         glob(f'{constants.DATASET_FOLDER}/{data_split}/1.Datasets/2.라벨링데이터/**/AD/*/AXL/*.json', recursive=True)]
        self.ann_path += [x.replace(os.sep, '/') for x in
                         glob(f'{constants.DATASET_FOLDER}/{data_split}/1.Datasets/2.라벨링데이터/**/FD/*/AXL/*.json', recursive=True)]
        self.ann_path += [x.replace(os.sep, '/') for x in
                         glob(f'{constants.DATASET_FOLDER}/{data_split}/1.Datasets/2.라벨링데이터/**/N/*/AXL/*.json', recursive=True)]

        self.cat_name = ['Tibia', 'Fibula', 'Talus', 'Calcaneus', 'MidFoot',
                         '1st Metatarsal', '2nd Metatarsal', '3rd Metatarsal', '4th Metatarsal', '5th Metatarsal']
        self.rle = False
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def load_data(self):
        data = []
        cls_n = {}
        for c in self.cat_name:
            cls_n.update({c: 0})
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
                if 'polyline_list' not in anno.keys():
                    continue
                try:
                    polyline = anno['polyline_list'][0]
                    seg = [[x['Y'], x['X']] for x in polyline['pos_list']]
                    cat = {
                        'category_id': self.cat_name.index(anno['preset_name']),
                        'segmentation': seg
                    }
                    cls_n[anno['preset_name']] += 1
                    _ann['anno'].append(cat)
                except KeyError as e:
                    print(f'Annotation file has invalid Key: {ann_path}')
                    print(e)
                    continue
                except ValueError as e:
                    # print(e)
                    # print(ann_path)
                    continue

            if len(_ann['anno']) > 0:
                data.append(_ann)
                data_id += 1

        print(f'End loading data. The number of data: {len(data)}')
        print('**************************************************')
        for k, v in cls_n.items():
            print(f'{k}: {v}')
        print('**************************************************')
        return data


if __name__ == '__main__':
    train_dataset = NIADataset(data_split='train')
    val_dataset = NIADataset(data_split='val')
    test_dataset = NIADataset(data_split='test')
