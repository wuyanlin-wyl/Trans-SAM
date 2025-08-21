import pandas as pd
import os
import torch as t
import numpy as np
import torchvision.transforms.functional as ff
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import model_utils.cfg as cfg
import random
import torchvision.transforms.functional as TF

class LabelProcessor:

    def __init__(self, file_path):

        self.colormap = self.read_color_map(file_path)

        self.cm2lbl = self.encode_label_pix(self.colormap)

    @staticmethod
    def read_color_map(file_path):
        pd_label_color = pd.read_csv(file_path, sep=',')
        colormap = []
        for i in range(len(pd_label_color.index)):
            tmp = pd_label_color.iloc[i]
            color = [tmp['r'], tmp['g'], tmp['b']]
            colormap.append(color)
        return colormap

    @staticmethod
    def encode_label_pix(colormap): 
        cm2lbl = np.zeros(256 ** 3)
        for i, cm in enumerate(colormap):
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        return cm2lbl

    def encode_label_img(self, img):

        data = np.array(img, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(self.cm2lbl[idx], dtype='int64')


class Dataset_train(Dataset):
    def __init__(self, file_path=[], crop_size=None):

        self.img_path = file_path[0]
        self.label_path = file_path[1]

        self.imgs = self.read_file(self.img_path)
        self.labels = self.read_file(self.label_path)

        self.crop_size = crop_size

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]

        img = Image.open(img).convert('RGB')
        label = Image.open(label).convert('RGB')

        img = img.resize((cfg.image_size, cfg.image_size))
        label = label.resize((cfg.image_size, cfg.image_size))

        img, label = self.img_transform(img, label)

        sample = {'img': img, 'label': label}

        return sample

    def __len__(self):
        return len(self.imgs)

    def read_file(self, path):

        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

    def center_crop(self, data, label, crop_size):

        data = ff.center_crop(data, crop_size)
        label = ff.center_crop(label, crop_size)
        return data, label

    def img_transform(self, img, label, pattern=None):

        label = np.array(label) 
        label = Image.fromarray(label.astype('uint8'))

        transform_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.GaussianBlur((25, 25), sigma=(0.001, 2.0)),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01
                ),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        )

        img = transform_img(img)
        label = label_processor.encode_label_img(label)
        label = t.from_numpy(label).unsqueeze(0)

        if random.uniform(0.0, 1.0) > 0.5:
            img = TF.hflip(img)
            label = TF.hflip(label)

        if random.uniform(0.0, 1.0) > 0.5:
            img = TF.vflip(img)
            label = TF.vflip(label)

        angle = random.uniform(-180.0, 180.0)
        h_trans = random.uniform(-256 / 8, 256 / 8)
        v_trans = random.uniform(-256 / 8, 256 / 8)
        scale = random.uniform(0.5, 1.5)
        shear = random.uniform(-22.5, 22.5)
        img = TF.affine(img, angle, (h_trans, v_trans), scale, shear)
        label = TF.affine(label, angle, (h_trans, v_trans), scale, shear)

        return img, label.squeeze()


class Dataset_val_test(Dataset):
    def __init__(self, file_path=[], crop_size=None):

        self.img_path = file_path[0]
        self.label_path = file_path[1]

        self.imgs = self.read_file(self.img_path)
        self.labels = self.read_file(self.label_path)

        self.crop_size = crop_size

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]

        img = Image.open(img).convert('RGB')
        label = Image.open(label).convert('RGB')

        img = img.resize((cfg.image_size, cfg.image_size))
        label = label.resize((cfg.image_size, cfg.image_size))

        img, label = self.img_transform(img, label)

        sample = {'img': img, 'label': label}

        return sample

    def __len__(self):
        return len(self.imgs)

    def read_file(self, path):

        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

    def center_crop(self, data, label, crop_size):

        data = ff.center_crop(data, crop_size)
        label = ff.center_crop(label, crop_size)
        return data, label

    def img_transform(self, img, label):

        label = np.array(label) 
        label = Image.fromarray(label.astype('uint8'))
        transform_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        img = transform_img(img)
        label = label_processor.encode_label_img(label)
        label = t.from_numpy(label)

        return img, label


class Dataset_predict(Dataset):
    def __init__(self, file_path=[], crop_size=None):
        self.img_path = file_path[0]

        self.imgs = self.read_file(self.img_path)

        self.crop_size = crop_size

    def __getitem__(self, index):
        img = self.imgs[index]

        img = Image.open(img).convert('RGB')

        img = img.resize((256, 256))

        img = self.img_transform(img)

        sample = {'img': img}

        return sample

    def __len__(self):
        return len(self.imgs)

    def read_file(self, path):

        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

    def center_crop(self, data, crop_size):

        data = ff.center_crop(data, crop_size)

        return data

    def img_transform(self, img):

        transform_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        img = transform_img(img)

        return img


def build_all_layer_point_grids(
        n_per_side: int, n_layers: int, scale_per_layer: int
):
    """Generates point grids for all crop layers."""
    points_by_layer = []
    for i in range(n_layers + 1):
        n_points = int(n_per_side / (scale_per_layer ** i))
        points_by_layer.append(build_point_grid(n_points))
    return points_by_layer


def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points


class Dataset_SAM(Dataset):
    def __init__(self, file_path=[], crop_size=None, point_num=8, image_size=512):

        self.img_path = file_path[0]
        self.label_path = file_path[1]

        self.imgs = self.read_file(self.img_path)
        self.labels = self.read_file(self.label_path)

        self.crop_size = crop_size
        self.image_size = image_size

        self.point_grids = build_all_layer_point_grids(
            point_num,
            0,
            1,
        )

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]

        img = Image.open(img).convert('RGB')
        label = Image.open(label).convert('RGB')


        img = img.resize((self.image_size, self.image_size))
        label = label.resize((self.image_size, self.image_size))

        img, label = self.img_transform(img, label)

        original_size = (img.shape[1], img.shape[2])

        points_for_image = self.point_grids[0] * original_size
        in_points = t.as_tensor(points_for_image)
        in_labels = t.ones(in_points.shape[0], dtype=t.int)

        sample = {'image': img,
                  'label': label,
                  'point_coords': in_points,
                  'point_labels': in_labels,
                  }

        return sample

    def __len__(self):
        return len(self.imgs)

    def read_file(self, path):

        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

    def center_crop(self, data, label, crop_size):

        data = ff.center_crop(data, crop_size)
        label = ff.center_crop(label, crop_size)
        return data, label

    def img_transform(self, img, label):

        label = np.array(label) 
        label = Image.fromarray(label.astype('uint8'))
        transform_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        img = transform_img(img)
        label = label_processor.encode_label_img(label)
        label = t.from_numpy(label)

        return img, label


label_processor = LabelProcessor(cfg.class_dict_path)
