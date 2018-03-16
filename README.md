# mask-rcnn-pytorch

## Build

```bash
./build_layers.sh
```

## Use
1. basic info in `common.py`, run to check
2. run `make_annotation.py`, do annotation & making train dataset
3. run `train.py` to train model
4. run `validation.py` to do validation
5. run `submit.py` to submit csv file

## data folder structure

```txt
data=DATA_DIR
    split=SPLIT_DIR
        split1
        ...
    image=IMAGE_DIR
        <stage1_train>
            <images>
                id.png
            <multi_masks>
                id.png
                id.npy
            <overlays>
                id.png
        <stage1_test>
            <images>
                id.png
    __download__
        stage1_train (origin)
        stage1_test  (origin)
```

## result folder structure

```txt
results
    model_name
        <checkpoint>
            <iter_model.pth>
            <configuration.pkl>
        <train>
        <backup>
        <sumbit>
            <overlays>
            <npys>
        <log.train.txt>
```
