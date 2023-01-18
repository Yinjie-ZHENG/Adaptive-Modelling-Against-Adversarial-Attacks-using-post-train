
import os
import warnings
import functools
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

import torch
import torchvision
import torch.optim as optim
from torchsummary import summary
from torch import sigmoid,softmax
from torch.utils.data import  DataLoader,Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
import shutil
from V4args import *
from utils.arg_utils import *
from utils.data_utils import *
from utils.algorithm_utils import *
from dataloder import load_dataset
from metrics import Accuracy_score, AverageMeter, accuracy, accuracy2
#from models.ClassicNetwork.ResNet import ResNet18
#from models.AttackNetwork.mnist_net import mnist_net
from torchattacks import PGD, FGSM
from torch.utils.data import Subset
import copy
from torchattacks import PGD, PGDwDelta




logger = get_logger(data_config.work_dir + '/exp.log')
shutil.copyfile("./V4args.py", data_config.work_dir+"/configs.py")

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def cal_accuracy(outputs, labels):
    _, predictions = torch.max(outputs, 1)
    # collect the correct predictions for each class
    correct = 0
    total = 0
    for label, prediction in zip(labels, predictions):
        if label == prediction:
            correct += 1
        total += 1
    return correct / total

def get_train_loaders_by_class(train_dataset, batch_size):
    indices_list = [[] for _ in range(data_config.num_class)]
    for i in range(len(train_dataset)):
        label = int(train_dataset[i][1])
        indices_list[label].append(i)
    dataset_list = [Subset(train_dataset, indices) for indices in indices_list]
    train_loader_list = [
        torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
        ) for dataset in dataset_list
    ]
    return train_loader_list

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std
norm_layer = Normalize(mean=data_config.normalized_cfg_mean, std=data_config.normalized_cfg_std)

def post_train(model, images, train_loader, train_loaders_by_class, args):
    #保证插入的images是单张
    assert len(images) == 1, "Post training algorithm only accepts test input of batch size 1"

    logger = logging.getLogger("eval")

    alpha = data_config.alpha #10/255
    epsilon = data_config.epsilon #8/255
    loss_func = nn.CrossEntropyLoss()

    device = torch.device('cuda')
    model = copy.deepcopy(model)
    # model.train()
    fix_model = copy.deepcopy(model)
    # attack_model = torchattacks.PGD(model, eps=(8/255)/std, alpha=(2/255)/std, steps=20)
    optimizer = torch.optim.SGD(lr=args.pt_lr,
                                params=model.parameters(),
                                momentum=0.9,
                                nesterov=True)
    images = images.detach()
    with torch.enable_grad():
        # find neighbour
        original_output = fix_model(images)
        original_class = torch.argmax(original_output).reshape(1)

        if args.neigh_method == 'targeted':
            # targeted attack to find neighbour
            min_target_loss = float('inf')
            max_target_loss = float('-inf')
            neighbour_delta = None
            for target_idx in range(10):
                if target_idx == original_class:
                    continue
                target = torch.ones_like(original_class) * target_idx

                attack_pgd_targeted = PGDwDelta(model, eps=epsilon, alpha=alpha, steps=20, random_start=False)
                attack_pgd_targeted._set_mode_targeted()
                adv_images_targeted, neighbour_delta_targeted = attack_pgd_targeted(images, original_class)
                target_output = fix_model(adv_images_targeted)

                target_loss = loss_func(target_output, target)
                if target_loss < min_target_loss:
                    min_target_loss = target_loss
                    neighbour_delta = neighbour_delta_targeted
                # print(int(target), float(target_loss))
        elif args.neigh_method == 'untargeted':
            attack_pgd = PGDwDelta(model, eps=epsilon, alpha=alpha, steps=20, random_start=False)
            neighbour_images, neighbour_delta = attack_pgd(images, original_class)



        #neighbour_images = neighbour_delta + images
        neighbour_output = fix_model(neighbour_images)
        neighbour_class = torch.argmax(neighbour_output).reshape(1)

        if original_class == neighbour_class:
            logger.info('original class == neighbour class')
            if args.pt_data == 'ori_neigh':
                return model, original_class, neighbour_class, None, None, neighbour_delta

        loss_list = []
        acc_list = []
        for _ in range(args.pt_iter):
            if args.pt_data == 'ori_neigh':
                original_data, original_label = next(iter(train_loaders_by_class[original_class]))
                neighbour_data, neighbour_label = next(iter(train_loaders_by_class[neighbour_class]))
            elif args.pt_data == 'ori_rand':
                original_data, original_label = next(iter(train_loaders_by_class[original_class]))
                neighbour_class = (original_class + random.randint(1, 10)) % 10
                neighbour_data, neighbour_label = next(iter(train_loaders_by_class[neighbour_class]))
            elif args.pt_data == 'train':
                original_data, original_label = next(iter(train_loader))
                neighbour_data, neighbour_label = next(iter(train_loader))
            else:
                raise NotImplementedError

            data = torch.vstack([original_data, neighbour_data]).to(device)
            label = torch.hstack([original_label, neighbour_label]).to(device)

            if args.pt_method == 'adv':
                # generate fgsm adv examples
                delta = (torch.rand_like(data) * 2 - 1) * epsilon  # uniform rand from [-eps, eps]
                noise_input = data + delta
                noise_input.requires_grad = True
                noise_output = model(noise_input)
                loss = loss_func(noise_output, label)  # loss to be maximized
                input_grad = torch.autograd.grad(loss, noise_input)[0]
                delta = delta + alpha * torch.sign(input_grad)
                delta.clamp_(-epsilon, epsilon)
                adv_input = data + delta
            elif args.pt_method == 'dir_adv':
                # use fixed direction attack
                if args.adv_dir == 'pos':
                    adv_input = data + 1 * neighbour_delta
                elif args.adv_dir == 'neg':
                    adv_input = data + -1 * neighbour_delta
                elif args.adv_dir == 'both':
                    directed_delta = torch.vstack([torch.ones_like(original_data).to(device) * neighbour_delta,
                                                    torch.ones_like(neighbour_data).to(device) * -1 * neighbour_delta])
                    adv_input = data + directed_delta
            elif args.pt_method == 'normal':
                adv_input = data
            else:
                raise NotImplementedError

            adv_output = model(adv_input.detach())

            loss = loss_func(adv_output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            defense_acc = accuracy2(adv_output, label)#cal_accuracy(adv_output, label)
            loss_list.append(loss)
            acc_list.append(defense_acc)
    return model, original_class, neighbour_class, loss_list, acc_list, neighbour_delta


logger.info("***********- ***********- READ DATA and processing-*************")
train_dataset, val_dataset = load_dataset(data_config)
train_loader = DataLoader(train_dataset, batch_size=data_config.batch_size, shuffle=True)
test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
train_loaders_by_class = get_train_loaders_by_class(train_dataset,data_config.batch_size)

logger.info("***********- loading model -*************")
if(len(data_config.gpus)==0):#cpu
    model = data_config.model_arch
    if data_config.load_from_pth is not None:
        checkpoint = torch.load(data_config.load_from_pth)
        model.load_state_dict(checkpoint)
    elif data_config.load_from_pkl is not None:
        model, _, _ = load_checkpoint(model=model, checkpoint_path=data_config.load_from_pkl)
    # load之后，开始加normalize层
    if data_config.normalized:
        model = nn.Sequential(norm_layer, model)
elif(len(data_config.gpus)==1):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(data_config.gpus[0])
    model = data_config.model_arch.cuda()
    if data_config.load_from_pth is not None:
        checkpoint = torch.load(data_config.load_from_pth)
        model.load_state_dict(checkpoint)
    elif data_config.load_from_pkl is not None:
        model, _, _ = load_checkpoint(model=model,checkpoint_path=data_config.load_from_pkl)
    # load之后，开始加normalize层
    if data_config.normalized:
        model = nn.Sequential(norm_layer, model).cuda()
else:#multi gpus
    gpus = ','.join(str(i) for i in data_config.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    model = data_config.model_arch.cuda()
    gpus = [i for i in range(len(data_config.gpus))]

    #model.load_state_dict(torch.load(model_path))
    if data_config.load_from_pth is not None:
        checkpoint = torch.load(data_config.load_from_pth)
        model.load_state_dict(checkpoint)
    elif data_config.load_from_pkl is not None:
        if data_config.normalized:
            model = nn.Sequential(norm_layer, model).cuda()
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
        model, _, _ = load_checkpoint(model=model, checkpoint_path=data_config.load_from_pkl)
    # load之后，开始加normalize层


    #model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
model.eval()

epsilon = 8/255#0.3
alpha = 2/255#1e-2
pgd_loss = 0
pgd_acc = 0
pgd_loss_post = 0
pgd_acc_post = 0
normal_loss = 0
normal_acc = 0
normal_loss_post = 0
normal_acc_post = 0
neighbour_acc = 0

pgd_loss_rep = AverageMeter()
pgd_acc_rep = AverageMeter()
pgd_loss_post_rep = AverageMeter()
pgd_acc_post_rep = AverageMeter()
normal_loss_rep = AverageMeter()
normal_acc_rep = AverageMeter()
normal_loss_post_rep = AverageMeter()
normal_acc_post_rep = AverageMeter()


n = 0





for i, (X, y) in enumerate(test_loader):
    n += y.size(0)
    X, y = X.cuda(), y.cuda()
    #pgd_delta = attack_pgd(model, X, y, epsilon, alpha, data_config.attack_iters, data_config.attack_restart).detach()
    atk_pgd = PGDwDelta(model, eps=epsilon, alpha=alpha, steps=20, random_start=False)
    _, pgd_delta = atk_pgd(X, y)


    logger.info("\n")
    # evaluate base model
    with torch.no_grad():
        output = model(X + pgd_delta)
        loss = F.cross_entropy(output, y)
        pgd_loss += loss.item() * y.size(0)
        pgd_acc += (output.max(1)[1] == y).sum().item()
        logger.info('Batch {}\tbase acc: {:.4f}'.format(i + 1, pgd_acc / n))
        pgd_loss_rep.update(loss.item(),y.size(0))
        acc_top1 = accuracy2(softmax(output, dim=-1).data, y) #softmax(output, dim=-1)是predicted
        pgd_acc_rep.update(acc_top1[0].item(), y.size(0))
        logger.info('Batch {}\tbase acc_rep: {:.4f}'.format(i + 1, pgd_acc_rep.avg))


    # evaluate post model against adv
    with torch.no_grad():
        post_model, original_class, neighbour_class, _, _, _ = post_train(model, X + pgd_delta, train_loader,
                                                                          train_loaders_by_class, data_config)
        # evaluate neighbour acc
        neighbour_acc += 1 if int(y) == int(original_class) or int(y) == int(neighbour_class) else 0
        logger.info('Batch {}\tneigh acc: {:.4f}'.format(i + 1, neighbour_acc / n))

        # evaluate prediction acc
        output = post_model(X + pgd_delta)
        loss = F.cross_entropy(output, y)
        pgd_loss_post += loss.item() * y.size(0)
        pgd_acc_post += (output.max(1)[1] == y).sum().item()


        logger.info('Batch {}\tadv acc (post): {:.4f}'.format(i + 1, pgd_acc_post / n))
        pgd_loss_post_rep.update(loss.item(),y.size(0))
        acc_top1 = accuracy2(softmax(output, dim=-1).data, y) #softmax(output, dim=-1)是predicted
        pgd_acc_post_rep.update(acc_top1[0].item(), y.size(0))
        logger.info('Batch {}\tbase acc_rep: {:.4f}'.format(i + 1, pgd_acc_post_rep.avg))

    # evaluate base model against normal
    with torch.no_grad():
        output = model(X)
        loss = F.cross_entropy(output, y)
        normal_loss += loss.item() * y.size(0)
        normal_acc += (output.max(1)[1] == y).sum().item()
        logger.info('Batch {}\tnormal acc: {:.4f}'.format(i + 1, normal_acc / n))

        normal_loss_rep.update(loss.item(),y.size(0))
        acc_top1 = accuracy2(softmax(output, dim=-1).data, y) #softmax(output, dim=-1)是predicted
        normal_acc_rep.update(acc_top1[0].item(), y.size(0))
        logger.info('Batch {}\tbase acc_rep: {:.4f}'.format(i + 1, normal_acc_rep.avg))

    # evaluate post model against normal
    with torch.no_grad():
        post_model, original_class, neighbour_class, _, _, _ = post_train(model, X, train_loader,
                                                                          train_loaders_by_class, data_config)
        output = post_model(X)
        loss = F.cross_entropy(output, y)
        normal_loss_post += loss.item() * y.size(0)
        normal_acc_post += (output.max(1)[1] == y).sum().item()
        logger.info('Batch {}\tnormal acc (post): {:.4f}'.format(i + 1, normal_acc_post / n))

        normal_loss_post_rep.update(loss.item(), y.size(0))
        acc_top1 = accuracy2(softmax(output, dim=-1).data, y)  # softmax(output, dim=-1)是predicted
        normal_acc_post_rep.update(acc_top1[0].item(), y.size(0))
        logger.info('Batch {}\tbase acc_rep: {:.4f}'.format(i + 1, normal_acc_post_rep.avg))





print("done")