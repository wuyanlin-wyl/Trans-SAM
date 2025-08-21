import torch as t
import torch.nn as nn
import torch.nn.functional as F
import model_utils.cfg as cfg
import numpy as np
import os
import math
import random
from torch.utils import data
from torch.optim.lr_scheduler import LambdaLR
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
from SAM.build_sam import model_sam
from model_utils.dataset_split import Dataset_train, Dataset_val_test
from model_utils.evalution_segmentation import eval_semantic_segmentation,get_dice
from sklearn.model_selection import train_test_split
from tqdm import tqdm

np.seterr(divide='ignore', invalid='ignore')

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False

set_seed(20)

device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

train = Dataset_train([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL])
val = Dataset_val_test([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL])


def split_ids(len_ids):
    train_q = 80
    train_size = int(round((train_q / 100) * len_ids))
    valid_size = int(round(((90-train_q) / 100) * len_ids))
    test_size = int(round((10 / 100) * len_ids))

    train_indices, test_indices = train_test_split(
        np.linspace(0, len_ids - 1, len_ids).astype("int"),
        test_size=test_size,
        random_state=42,
    )

    train_indices, val_indices = train_test_split(
        train_indices, test_size=valid_size, random_state=42
    )

    return train_indices, test_indices, val_indices, train_q

input_data_len = len(sorted(os.listdir(cfg.TRAIN_ROOT)))
train_indices, test_indices, val_indices, train_q  = split_ids(input_data_len)

train = data.Subset(train, train_indices)
val = data.Subset(val, val_indices)

train_data = DataLoader(train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=8,pin_memory=True, prefetch_factor=2)
val_data = DataLoader(val, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=8,pin_memory=True, prefetch_factor=2)

image_size = cfg.image_size
sam = model_sam(image_size=image_size,num_classes=1).to(device)


criterion = nn.NLLLoss().to(device)
optimizer = optim.AdamW(sam.parameters(), lr=cfg.lr, weight_decay=1e-8)  #0.0001  Adam


total_epochs = cfg.EPOCH_NUMBER
warmup_epochs = 10
cosine_decay_epochs = total_epochs - warmup_epochs


def warmup_lr(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    else:
        return 1


def cosine_decay_lr(epoch):
    return 0.5 * (1 + math.cos((epoch - warmup_epochs) / cosine_decay_epochs * math.pi))

def combined_lr(epoch):
    return warmup_lr(epoch) * cosine_decay_lr(epoch)

scheduler = LambdaLR(optimizer, lr_lambda=combined_lr)

def train(model):
    best = [0]
    best_epoch = 0
    iter_train = 0
    iter_val = 0

    for epoch in range(cfg.EPOCH_NUMBER):
        print("Start the {} epoch of model training".format(epoch + 1))

        train_loss = 0
        train_acc = 0
        train_miou = 0
        train_class_acc = 0
        total = 0

        net = model.train()

        train_time = datetime.now() 
        train_bar = tqdm(train_data, colour='blue')

        for i, sample in enumerate(train_bar):
            iter_train += 1
            labels = sample['label']
            total += labels.size(0)

            img_data = Variable(sample['img'].to(device))
            img_label = Variable(sample['label'].to(device))

            out = net(img_data,image_size = image_size)
            out = F.log_softmax(out, dim=1)

            preout = out.max(dim=1)[1].data.cpu().numpy()
            gtout = img_label.data.cpu().numpy()

            DICE_loss = 1-get_dice(preout, gtout)

            BCE_loss = criterion(out, img_label)

            loss = BCE_loss + DICE_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     cfg.EPOCH_NUMBER,
                                                                     loss.item())

            preout = out.max(dim=1)[1].data.cpu().numpy()
            gtout = img_label.data.cpu().numpy()

            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = img_label.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metrix = eval_semantic_segmentation(pre_label, true_label, preout, gtout)
            train_acc += eval_metrix['mean_class_accuracy']
            train_miou += eval_metrix['miou']
            train_class_acc += eval_metrix['class_accuracy']


        scheduler.step()

        for group in optimizer.param_groups:
            print('The current learning rate is:', group['lr'])


        metric_description = '|Train Acc|: {:.5f}|Train Mean IU|: {:.5f}\n|Train_class_acc|:{:}'.format(
            train_acc / len(train_data),
            train_miou / len(train_data),
            train_class_acc / len(train_data)
        )
        print(metric_description)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - train_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = 'Train_Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
        print(time_str)


        if (epoch + 1) % 1 == 0:
            print("Start the {} epoch of model verification".format(epoch + 1))
            net = model.eval()

            eval_loss = 0
            eval_acc = 0
            eval_miou = 0
            eval_class_acc = 0
            total_val = 0

            val_time = datetime.now()
            val_bar = tqdm(val_data, colour='red')
            with t.no_grad():
                for j, sample in enumerate(val_bar):
                    iter_val += 1
                    labels_ = sample['label']
                    total_val += labels_.size(0)

                    valImg = Variable(sample['img'].to(device))
                    valLabel = Variable(sample['label'].long().to(device))

                    out = net(valImg,image_size = image_size)

                    out = F.log_softmax(out, dim=1)
                    val_loss = criterion(out, valLabel)
                    eval_loss = val_loss.item() + eval_loss

                    val_bar.desc = "val iteration[{}/{}] loss:{:.3f}".format(j + 1,
                                                                             len(val_data),
                                                                             val_loss.item())

                    preout = out.max(dim=1)[1].data.cpu().numpy()
                    gtout = valLabel.data.cpu().numpy()

                    pre_label = out.max(dim=1)[1].data.cpu().numpy()
                    pre_label = [i for i in pre_label]

                    true_label = valLabel.data.cpu().numpy()
                    true_label = [i for i in true_label]

                    eval_metrics = eval_semantic_segmentation(pre_label, true_label, preout, gtout)
                    eval_acc = eval_metrics['mean_class_accuracy'] + eval_acc
                    eval_miou = eval_metrics['miou'] + eval_miou
                    eval_class_acc = eval_metrics['class_accuracy'] + eval_class_acc

                val_str = (
                    '|Valid Loss|: {:.5f} \n|Valid Acc|: {:.5f} \n|Valid Mean IU|: {:.5f} \n|Valid Class Acc|:{:}'.format(
                        eval_loss / len(val_data),
                        eval_acc / len(val_data),
                        eval_miou / len(val_data),
                        eval_class_acc / len(val_data)))
                print(val_str)

                cur_time = datetime.now()
                h, remainder = divmod((cur_time - val_time).seconds, 3600)
                m, s = divmod(remainder, 60)
                time_str = 'Val_Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
                print(time_str)

                if max(best) <= eval_miou / len(val_data):
                    best.append(eval_miou / len(val_data))
                    t.save(net.state_dict(), './weight/{}.pth'.format(epoch + 1))
                    best_epoch = epoch + 1
                print("The maximum IOU of the current model is {:.5f}, and the corresponding number of epochs is {}".format(best[-1], best_epoch))



if __name__ == "__main__":
    train(sam)
