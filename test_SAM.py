import torch as t
import torch.nn.functional as F
import model_utils.cfg as cfg
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model_utils.evalution_segmentation import eval_semantic_segmentation
from model_utils.dataset_split import Dataset_val_test
from sklearn.model_selection import train_test_split
from torch.utils import data
from SAM.build_sam import model_sam
import time
import numpy as np
from model_utils import calculation_network_model_parameters as tj
from tqdm import tqdm
import os

device = t.device('cuda:0') if t.cuda.is_available() else t.device('cpu')

BATCH_SIZE = 6

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
        train_indices, test_size=valid_size, random_state=42
    )

    return train_indices, test_indices, val_indices

input_data_len = len(sorted(os.listdir(cfg.TRAIN_ROOT)))
_, test_indices, _ = split_ids(input_data_len)

test = data.Subset(test, test_indices)

test_data = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


image_size = cfg.image_size
net = model_sam(image_size=image_size,num_classes=1).to(device)

net.eval()
tj.model_structure(net)


def get_last_ten_files(folder_path):

    files = sorted(os.listdir(folder_path), key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))

    last_ten_files = files[-10:]

    return last_ten_files


folder_path = './weight/'


last_ten_files = get_last_ten_files(folder_path)


results = []

test_bar = tqdm(last_ten_files, colour='blue')
for file_name in test_bar:
    net.load_state_dict(t.load('./weight/{}'.format(file_name)))

    train_acc = 0
    train_miou = 0
    train_class_acc = 0
    train_mpa = 0
    error = 0
    JS = 0
    jaccard = 0
    DC = 0
    SP = 0
    SE = 0
    PC = 0
    RE = 0
    RVD = 0
    VOE = 0

    with t.no_grad():
        for i, sample in enumerate(test_data):
            
            data = Variable(sample['img']).to(device)
            label = Variable(sample['label']).to(device)
            out = net(data, image_size=image_size)
            
            out = F.log_softmax(out, dim=1)

            preout = out.max(dim=1)[1].data.cpu().numpy()
            gtout = label.data.cpu().numpy()

            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = label.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metrics = eval_semantic_segmentation(pre_label, true_label, preout, gtout)
            train_acc = eval_metrics['mean_class_accuracy'] + train_acc
            train_miou = eval_metrics['miou'] + train_miou

            JS = eval_metrics['JS'] + JS
            DC = eval_metrics['DC'] + DC
            SP = eval_metrics['SP'] + SP
            SE = eval_metrics['SE'] + SE
            PC = eval_metrics['PC'] + PC
            RE = eval_metrics['RE'] + RE
            RVD = eval_metrics['RVD'] + RVD
            VOE = eval_metrics['VOE'] + VOE

            if len(eval_metrics['class_accuracy']) < 2:
                eval_metrics['class_accuracy'] = 0
                train_class_acc = train_class_acc + eval_metrics['class_accuracy']
                error += 1
            else:
                train_class_acc = train_class_acc + eval_metrics['class_accuracy']


    result_dict = {
        'file_name': file_name,
        'JS': JS / (len(test_data) - error),
        'DC': DC / (len(test_data) - error),
        'SP': SP / (len(test_data) - error),
        'SE': SE / (len(test_data) - error),
        'PC': PC / (len(test_data) - error),
        'RE': RE / (len(test_data) - error),
        'RVD': RVD / (len(test_data) - error),
        'VOE': VOE / (len(test_data) - error),
        'test_acc': train_acc / (len(test_data) - error),
        'test_miou': train_miou / (len(test_data) - error),
        'test_class_acc': train_class_acc / (len(test_data) - error),
    }
    results.append(result_dict)


sorted_results = sorted(results, key=lambda x: x['DC'], reverse=True)


print("All results are sorted in descending order by DICE value:")
for result in sorted_results:
    print(f"文件名: {result['file_name']}, ", end='')
    for key, value in result.items():
        if key != 'file_name':

            if isinstance(value, np.ndarray):
                value = value.tolist()

            if isinstance(value, list):
                print(f"{key}: ", end='')
                for item in value:
                    print(f"{item:.5f}, ", end='')
            else:

                print(f"{key}: {value:.5f}, ", end='')
    print()