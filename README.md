# Installation
To use this code, ensure you have the following dependencies installed:  
Python 3.9.16  
PyTorch  
numpy  
torchvision  
tqdm  
argparse  
matplotlib  
json  
thop  
tensorboard  
torch  
easyfsl

# Datasets
The following datasets are used in this repository:
**CUB_dataset**: This dataset includes several modifications and classifier comparisons to evaluate the performance of CenterLiteNet.
**FC100_dataset**: Multiple configurations and classifiers are tested on this dataset.
**MINI_dataset**: Similar comparisons are made on this dataset, focusing on different modifications and classifiers.
Additionally, a custom-built **kiwifruit pest dataset** is included, allowing the model to be tested on real-world agricultural data. The dataset focuses on pest species that affect kiwifruit crops.
# Usage
python CenterLiteNet_cifar.py --lambda_center 0.001 --shot 1 --way 5 > 1s5w_cub_CenterLiteNet.log 2>&1 & 

