import torch
#import openseespy.opensees as ops
from torch_geometric.data import Data

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Running CPU-only")
'''
ops.wipe()
ops.model("basic", "-ndm", 2, "-ndf", 3)

data = Data()
print("OpenSeesPy + PyTorch Geometric working")
'''
'''
--------------------    Installation     --------------------
1. Git Clone Repository
-- Gpu --
conda env export > environment_gpu.yml
pip freeze > requirements_gpu.txt
2. Conda env create -f environment_gpu.yml
3. Conda activate OpPy
-- Cpu --
4. Conda env activate -f environment_cpu.yml
5. Run Verification 

pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install openseespy
pip install numpy scipy pandas matplotlib scikit-learn tqdm jupyter
pip install opsvis
pip install openseespywin
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip uninstall pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -y
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/nightly/torch-2.8.0+cu128.html