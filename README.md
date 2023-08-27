# Unet-segmentation

## Install

```shell
pip install -r requirements.txt
```

## Data prepare
```shell
python3 tools/data_preprocessing.py --img_path [IMG_DIR] --anno_path [ANN_DIR] --output_path [OUPUT_FILE_PATH] 
```

## Model train
```shell
python3 train.py
```

## Model evaluate
```shell
python3 eval.py --checkpoint [MODEL_FILE_PATH]
```
