cd net/lib/box/nms/torch_nms/src/
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_60
cd ..
python build.py

cd ../../../../layer/roi_align_pool_tf/src/
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_60
cd ..
python build.py

cd ../../lib/box/nms/cython_nms/
python setup.py build_ext --inplace

cd ../gpu_nms/
python setup.py build_ext --inplace
mv *.so gpu_nms.so

cd ../../overlap/cython_overlap/
python setup.py build_ext --inplace
mv *.so cython_box_overlap.so
