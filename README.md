# mask-rcnn-pytorch

## Build

```bash
cd mask-rcnn-resnet50-ver-01/net/lib/box/nms/torch_nms/src/
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_60
cd ..
python build.py

cd ../../../roi_align_pool_tf/src/
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_60
cd ..
python build.py

cd ../box/nms/cython_nms/
python setup.py build_ext --inplace

cd ../gpu_nms/
python setup.py build_ext --inplace
# move .so to this folder?

cd ../../overlap/cython_overlap/
python setup.py build_ext --inplace
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
