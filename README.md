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

---

## 安装shadowsocks-libev 客户端

```bash
apt-get install software-properties-common -y
add-apt-repository ppa:max-c-lv/shadowsocks-libev -y
apt-get update
apt install shadowsocks-libev
```

编辑配置文件:
```
vim /etc/shadowsocks-libev/config.json

{  
 "server":"ss5.gogosu.xyz",  
 "server_port":64220,  
 "local_port":1080,  
 "password":"j36EWk",  
 "timeout":60,  
 "method":"aes-256-gcm",   
}
```
启动
```
systemctl start shadowsocks-libev-local@config
```

## 安装proxychains-ng
```
git clone https://github.com/rofl0r/proxychains-ng.git
cd proxychains-ng
./configure --prefix=/usr --sysconfdir=/etc
make
make install
make install-config
```
打开配置文件
```
vim /etc/proxychains.conf
```
把最后一行改成
```
socks5 127.0.0.1 1080
```
