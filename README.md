# Kaggle 2018 Data Science Bowl

38th Solution of [Kaggle 2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018). LB 0.565  

Our team was removed from LB due to [submission issues](https://www.kaggle.com/c/data-science-bowl-2018/discussion/55062#latest-318477).

1. We use pytorch mask-rcnn as our base model. The code is mainly from [Heng CherKeng](https://www.kaggle.com/c/data-science-bowl-2018/discussion/49692#latest-315307)
2. We changed anchor boxes to fit long/thin nuclei
3. We use special cluster based filter to get rid of small false negative samples


## Build
Requirements:
1. pytorch 0.4
2. python opencv 3
3. Anaconda 3

```bash
./build_layers.sh
```

## Use
1. Run `common.py` to check environment. modify `ROOT_DIR` to where you put `data` and `result` folder
2. `configuration.py` training configurations
3. run `make_annotation.py`, do annotation & making train dataset
4. run `train.py` to train model
5. run `validation.py` to do validation
6. run `submit.py` to submit csv file

## data folder structure
folder labeded `<>` is generated automatically, other folders need manually set
```txt
data
    splits 存放训练集/测试集分割
        split1
        ...
    <stage1_train>
        <images>
            id.png
        <multi_masks> 存放了mask, 包括图片和npy
            id.png
            id.npy
        <overlays>  存放了mask和原图拼接的结果, 仅供观察
            id.png
    <stage1_test>
        <images>
            id.png
    source 存放下载的数据集
        stage1_train
        stage1_test
```

## result folder structure

```txt
results
    <model_name>
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
