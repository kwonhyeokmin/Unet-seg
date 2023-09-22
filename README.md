# Unet-segmentation

## Install

```shell
pip install -r requirements.txt
```

## Data prepare (for uwmgi dataset)
```shell
python3 tools/data_preprocessing.py --img_path [IMG_DIR] --anno_path [ANN_DIR] --output_path [OUPUT_FILE_PATH] 
```

## Model train
```shell
python3 train.py --dataset [DATASET_NAME] --cfg [CONFIG_FILE_PATH]
```

## Model evaluate
```shell
python3 eval.py --checkpoint [MODEL_FILE_PATH] --dataset [DATASET_NAME] --cfg [CONFIG_FILE_PATH]
```
