# mask-rcnn-pytorch

## Build

```bash
./build_layers.sh
```

## Use
1. 运行`common.py`检查环境. 修改`ROOT_DIR`为存放`data`和`result`的文件夹路径
2. `configuration.py`保存了所有训练配置
3. run `make_annotation.py`, do annotation & making train dataset
4. run `train.py` to train model
5. run `validation.py` to do validation
6. run `submit.py` to submit csv file

## data folder structure
标记`<>`的文件夹是自动生成的, 其他文件夹都要手动创建
```txt
data=DATA_DIR
    split=SPLIT_DIR 存放训练集/测试集分割
        split1
        ...
    image=IMAGE_DIR 存放图片
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
    __download__
        stage1_train 下载训练集放这里
        stage1_test  下载测试集放这里
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
