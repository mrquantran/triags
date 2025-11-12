# Download preprocessed dataset
 
# Installation
1. create an environment 
```
conda create -n triags python=3.10
```

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
``

Install the cuda toolkit from https://developer.nvidia.com/cuda-12-6-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=runfile_local

Specify the path
```
export PATH=/usr/local/cuda-12.6/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH}
```

```
pip install submodules/diff-gaussian-rasterization
pip install git+https://github.com/camenduru/simple-knn
```

2. 

