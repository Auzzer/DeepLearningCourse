import torch
print(f"TorchVersion: {torch.__version__}")

# Trainning Set

batch_size = 64
epochs = 100
lr = 3e-5
gamma = 0.7
seed = 1234
import numpy as np
import os
import random
def seed_everything(seed):
    """
    seed:种子数
    对所有随机设置种子数
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed)

os.makedirs('data', exist_ok=True)

train_dir = 'data/train'
test_dir = 'data/test'

import glob
train_list = glob.glob(os.path.join(train_dir, '*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

print(f"Train Data Num: {len(train_list)}")
print(f"Test Data Num: {len(test_list)}")


labels = [path.split('/')[-1].split('/')[0] for path in train_list]


import matplotlib.pyplot as plt
random_idx = np.random.randint(1, len(train_list), size = 9)
fig, axes = plt.subplots(3, 3, figsize=(16, 12))


from PIL import Image


for idx, ax in enumerate(axes.reval()):
    img = Image.open(train_list[idx])
    ax.set_title(labels[idx])
    ax.imshow(img)

## 划分测试训练集合
from sklearn.model_selection import train_test_split
train_list, valid_list = train_test_split(train_list,
                                          test_size = 0.2,
                                          stride = labels,
                                          random_state=seed)
print(f"Train Data Num: {len(train_list)}")
print(f" Validation Data Num: {len(valid_list)}")
print(f"Test Data Num: {len(test_list)}")

"""
数据预处理部分
注意：如果是在测试Robust的时候，测试集不要使用增加遮挡和高斯模糊
"""
from torchvision import transforms, datasets
train_transforms = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
                                      ])

val_transforms = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
                                      ])

test_transforms = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
                                      ])

# 加载数据
from torch.utils.data import  DataLoader, Dataset
class CatDog(Dataset):
    def __init__(self, file_list, transform = None):
        self.file_list = file_list
        self.transform  = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transfomed = self.transform(img)

        label = img_path.split("/")[-1].split("/")[0]
        label = 1 if label == "dog" else 0

        return img_transfomed, label

train_data = CatDog(train_list, transform = train_transforms)
valid_data = CatDog(valid_list, transform = val_transforms)
test_data = CatDog(test_list, transform = test_transforms)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

print(f"check the TrainSet loader: {len(train_data), len(train_loader)}")

print(f"check the ValidSet loader: {len(valid_data), len(valid_loader)}")


###训练前准备
print("开始训练检查")
print("检查GPU")
device = torch.device("cuda: 0" if(torch.cuda.is_available()) else "cpu")
print(device)
print("torch.cuda.get_device_name(0)" if(torch.cuda.is_available()) else "No GPU Availale")


################################定义网络################################################
from linformer import Linformer
import time
start_time = time.time()
efficient_transformer = Linformer(
    dim=128,
    seq_len=49+1, # 7x7 patches +1
    depth=12,
    heads = 8,
    k=64
)

import ViT
model = ViT(
    dim=128,
    image_size=224,
    patch_size=32,
    num_classes = 2,
    transformer = efficient_transformer,
    channels=3,
    depth = 12,
    heads = 8,
    mlp_dim = 128
).to(device)


