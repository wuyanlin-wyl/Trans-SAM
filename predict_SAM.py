import pandas as pd
import numpy as np
import os
import torch as t
import torch.nn.functional as F
import model_utils.cfg as cfg
from torch.utils.data import DataLoader
from PIL import Image
from model_utils.dataset_split import Dataset_val_test
from sklearn.model_selection import train_test_split
from torch.utils import data
from SAM.build_sam import model_sam

device = t.device('cuda:0') if t.cuda.is_available() else t.device('cpu')

test = Dataset_val_test([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL])

def split_ids(len_ids):
    train_size = int(round((80 / 100) * len_ids))
    valid_size = int(round((10 / 100) * len_ids))
    test_size = int(round((10 / 100) * len_ids))

    train_indices, test_indices = train_test_split(
        np.linspace(0, len_ids - 1, len_ids).astype("int"),
        test_size=test_size,
        random_state=42,
    )

    train_indices, val_indices = train_test_split(
        train_indices, test_size=test_size, random_state=42
    )

    return train_indices, test_indices, val_indices

input_data_len = len(sorted(os.listdir(cfg.TRAIN_ROOT)))
_, test_indices, _ = split_ids(input_data_len)

test = data.Subset(test, test_indices)

test_data = DataLoader(test, batch_size=1, shuffle=False, num_workers=0)

image_size = cfg.image_size
net = model_sam(image_size=image_size,num_classes=1).to(device)

model_name = "SAM"

net.load_state_dict(t.load('./weight/45.pth'))
net.eval()

pd_label_color = pd.read_csv('./model_utils/class_dict.csv', sep=',')
name_value = pd_label_color['name'].values
num_class = len(name_value)
colormap = []
for i in range(num_class):
    tmp = pd_label_color.iloc[i]
    color = [tmp['r'], tmp['g'], tmp['b']]
    colormap.append(color)

cm = np.array(colormap).astype('uint8')

dataset =cfg.dataset
dir = "./experiment/segmentation_result/" + dataset +"~"+ model_name + "/"

if not os.path.exists(dir):
    os.makedirs(dir)
    print("Directory '%s' created successfully." % dir)
else:
    print("Directory '%s' already exists." % dir)

with t.no_grad():
    for i, sample in enumerate(test_data):

        valImg = sample['img'].to(device)
        out = net(valImg)
        out = F.log_softmax(out, dim=1)
        pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
        pre = cm[pre_label]
        pre1 = Image.fromarray(pre)
        id = "{0:05d}".format(i + 1)
        pre1.save(dir + id + '.png')

    print('Output the segmentation map {} sheets'.format(len(test_data)))
