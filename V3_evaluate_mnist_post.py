# -*- coding:UTF-8 -*-
"""
training classifying task with CNNs

"""

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
from V3args import *
from utils.arg_utils import *
from utils.data_utils import *
from utils.algorithm_utils import *
from dataloder import load_dataset
from metrics import Accuracy_score, AverageMeter, accuracy, accuracy2
#from models.ClassicNetwork.ResNet import ResNet18
#from models.AttackNetwork.mnist_net import mnist_net
from torchattacks import PGD, FGSM



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



#***********- Hyper Arguments-*************
warnings.filterwarnings("ignore")
logger = get_logger(data_config.work_dir + '/exp.log')
shutil.copyfile("./V3args.py", data_config.work_dir+"/configs.py")

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device=torch.device(device_name)
if data_config.rand_seed>0:
    init_rand_seed(data_config.rand_seed)


logger.info("***********- ***********- READ DATA and processing-*************")

train_dataset,val_dataset = load_dataset(data_config)

#x,y= train_dataset[0]
#logger.info('input:{},lable:{}'.format(x.size(),y))#[3, 224, 224]) 0

logger.info("***********- loading model -*************")
if(len(data_config.gpus)==0):#cpu
    model = data_config.model_arch.cuda()

    # load之后，开始加normalize层
    if data_config.normalized:
        model = nn.Sequential(norm_layer, model).cuda()
elif(len(data_config.gpus)==1):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(data_config.gpus[0])
    model = data_config.model_arch.cuda()

    # load之后，开始加normalize层
    if data_config.normalized:
        model = nn.Sequential(norm_layer, model).cuda()
else:#multi gpus
    gpus = ','.join(str(i) for i in data_config.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    model = data_config.model_arch.cuda()


    gpus = [i for i in range(len(data_config.gpus))]

    #train时候不用load
    #model.load_state_dict(torch.load(model_path))
    # load之后，开始加normalize层
    if data_config.normalized:
        model = nn.Sequential(norm_layer, model).cuda()
    model = torch.nn.DataParallel(model, device_ids=gpus)#.cuda()





optimizer = eval(data_config.optimizer)(model.parameters(),**data_config.optimizer_parm)
scheduler = eval(data_config.scheduler)(optimizer,**data_config.scheduler_parm)
loss_f=eval(data_config.loss_f)()
loss_dv=eval(data_config.loss_dv)()
#loss_fn = eval(data_config.loss_fn)()


#***********- VISUALIZE -*************
# #tensorboard --logdir=<your_log_dir>
writer = SummaryWriter('runs/'+data_config.writer_log)
# # get some random training images
# images, labels = next(iter(train_dataset))cd
# images=torch.unsqueeze(images.permute(2,0,1),0).cuda()
# writer.add_graph(model, images)
# writer.close()





#***********- trainer -*************
class trainer:
    def __init__(self, loss_f,loss_dv, model, optimizer, scheduler, config):
        self.loss_f = loss_f
        self.loss_dv = loss_dv
        #self.loss_fn = loss_fn
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config



    def batch_train(self, batch_imgs, batch_labels, epoch):
        self.model.train()
        predicted = self.model(batch_imgs)
        loss =self.myloss(predicted, batch_labels)
        predicted = softmax(predicted, dim=-1)
        del batch_imgs, batch_labels
        return loss, predicted

    def train_epoch(self, loader, warmup_scheduler, epoch):
        self.model.train()
        tqdm_loader = tqdm(loader)
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        logger.info("\n************Training*************")
        for batch_idx, (imgs, labels) in enumerate(tqdm_loader):
            # print("data",imgs.size(), labels.size())#[128, 3, 32, 32]) torch.Size([128]
            if (len(data_config.gpus) > 0):
                imgs, labels = imgs.cuda(), labels.cuda()
            # print(self.optimizer.param_groups[0]['lr'])
            if data_config.attack == 'none':
                pass
            elif data_config.attack == 'fgsm':
                train_attack = FGSM(self.model, eps=data_config.epsilon)
                imgs = train_attack(imgs, labels)
                imgs = imgs.cuda()
            elif data_config.attack == 'pgd':
                train_attack = PGD(self.model, eps=data_config.epsilon, alpha=data_config.alpha, steps=data_config.attack_iters, random_start=True)

                imgs = train_attack(imgs, labels)
                imgs = imgs.cuda()
            else:
                logger.info("The attack {} is not yet implemented!".format(data_config.attack))

            loss, predicted = self.batch_train(imgs, labels, epoch)
            losses.update(loss.item(), imgs.size(0))
            # print(predicted.size(),labels.size())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #self.scheduler.step()

            err1, err5 = accuracy(predicted.data, labels, topk=(1, 5))
            top1.update(err1.item(), imgs.size(0))
            top5.update(err5.item(), imgs.size(0))

            tqdm_loader.set_description('Training: loss:{:.4}/{:.4} lr:{:.4} err1:{:.4} err5:{:.4}'.
                                        format(loss, losses.avg, self.optimizer.param_groups[0]['lr'],top1.avg, top5.avg))
            if epoch <= data_config.warm:
                warmup_scheduler.step()
            # if batch_idx%1==0:
            #     break
        return top1.avg, top5.avg, losses.avg

    def valid_fgsm(self, loader):
        self.model.eval()
        # tqdm_loader = tqdm(loader)
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        logger.info("\n************Evaluation on fgsm*************")
        attack = FGSM(self.model, eps=data_config.epsilon)

        for batch_idx, (batch_imgs, batch_labels) in enumerate(loader):
            if (len(data_config.gpus) > 0):
                batch_imgs, batch_labels = batch_imgs.cuda(), batch_labels.cuda()  # batch_imgs.to(device), batch_labels.to(device)#batch_imgs.cuda(), batch_labels.cuda()

            #delta = attack_fgsm(model, batch_imgs, batch_labels, data_config.epsilon)

            adv_images = attack(batch_imgs, batch_labels)

            with torch.no_grad():

                predicted = self.model(adv_images)
                loss = self.myloss(predicted, batch_labels).detach().cpu().numpy()
                predicted = softmax(predicted, dim=-1)
                losses.update(loss.item(), batch_imgs.size(0))

                err1, err5 = accuracy(predicted.data, batch_labels, topk=(1, 5))
                top1.update(err1.item(), batch_imgs.size(0))
                top5.update(err5.item(), batch_imgs.size(0))
        logger.info("\nval_loss:{:.4f} | acc1:{:.4f} | acc5:{:.4f}".format(losses.avg, 100-top1.avg, 100-top5.avg))

        return top1.avg, top5.avg, losses.avg

    def valid_pgd(self, loader):
        self.model.eval()
        # tqdm_loader = tqdm(loader)
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        logger.info("\n************Evaluation on pgd*************")
        attack = PGD(self.model, eps=data_config.epsilon, alpha=data_config.alpha, steps=data_config.attack_iters, random_start=True)


        for batch_idx, (batch_imgs, batch_labels) in enumerate(loader):
            if (len(data_config.gpus) > 0):
                batch_imgs, batch_labels = batch_imgs.cuda(), batch_labels.cuda()  # batch_imgs.to(device), batch_labels.to(device)#batch_imgs.cuda(), batch_labels.cuda()
            for _ in range(data_config.restarts):
                #logger.info("\n************Evaluation on pgd,restart_{}*************".format(_))
                adv_images = attack(batch_imgs, batch_labels)
                with torch.no_grad():

                    predicted = self.model(adv_images)
                    loss = self.myloss(predicted, batch_labels)#.detach().cpu().numpy()
                    predicted = softmax(predicted, dim=-1)
                    losses.update(loss.item(), batch_imgs.size(0))

                    err1, err5 = accuracy(predicted.data, batch_labels, topk=(1, 5))
                    top1.update(err1.item(), batch_imgs.size(0))
                    top5.update(err5.item(), batch_imgs.size(0))
        logger.info("\nval_loss:{:.4f} | err1:{:.4f} | err5:{:.4f}".format(losses.avg, 100-top1.avg, 100-top5.avg))

        return top1.avg, top5.avg, losses.avg

    def valid_epoch(self, loader):
        self.model.eval()
        # tqdm_loader = tqdm(loader)
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        logger.info("\n************Evaluation*************")
        for batch_idx, (batch_imgs, batch_labels) in enumerate(loader):
            with torch.no_grad():
                if (len(data_config.gpus) > 0):
                    batch_imgs, batch_labels = batch_imgs.cuda(), batch_labels.cuda()#batch_imgs.to(device), batch_labels.to(device)#batch_imgs.cuda(), batch_labels.cuda()
                predicted = self.model(batch_imgs)
                loss = self.myloss(predicted, batch_labels).detach().cpu().numpy()
                predicted = softmax(predicted, dim=-1)
                losses.update(loss.item(), batch_imgs.size(0))

                err1, err5 = accuracy(predicted.data, batch_labels, topk=(1, 5))
                top1.update(err1.item(), batch_imgs.size(0))
                top5.update(err5.item(), batch_imgs.size(0))
        logger.info("\nval_loss:{:.4f} | acc1:{:.4f} | acc5:{:.4f}".format(losses.avg, 100-top1.avg, 100-top5.avg))

        return top1.avg, top5.avg, losses.avg

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr=data_config.lr
        if data_config.dataset.startswith('cifar'):
            # lr = data_config.lr * (0.1 ** (epoch // (data_config.epochs * 0.3))) * (0.1 ** (epoch // (data_config.epochs * 0.75)))
            if epoch < 60:
                lr = data_config.lr
            elif epoch < 120:
                lr = data_config.lr * 0.2
            elif epoch < 160:
                lr = data_config.lr * 0.04
            else:
                lr = data_config.lr * 0.008
        elif data_config.dataset == ('imagenet'):
            if data_config.epochs == 300:
                lr = data_config.lr * (0.1 ** (epoch // 75))
            else:
                lr = data_config.lr * (0.1 ** (epoch // 30))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def myloss(self,predicted,labels):
        # print(predicted.size(),labels.size())#[128, 10]) torch.Size([128])
        loss = self.loss_f(predicted,labels)
        return loss

    def run(self, train_loder, val_loder,model_path, start=0):
        best_err1, best_err5, best_fgsmerr1, best_fgsmerr5, best_pgderr1, best_pgderr5 = 100,100,100,100,100,100
        start_epoch=start
        top_score = np.ones([5, 3], dtype=float)*100
        top_score5 = np.ones(5, dtype=float) * 100
        iter_per_epoch = len(train_loder)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * data_config.warm)

        for e in range(self.config.epochs):
            e=e+start_epoch+1
            logger.info("------model:{}----Epoch: {}--------".format(self.config.model_save_name, e))
            if e > data_config.warm:
                self.scheduler.step(e)
                # adjust_learning_rate(self.optimizer,e,data_config.model_type)
            # torch.cuda.empty_cache()
            _, _, train_loss = self.train_epoch(train_loder,warmup_scheduler,e)
            err1, err5, val_loss = self.valid_epoch(val_loder)
            logger.info("\nvalid_epoch | val_loss:{:.4f} | err1:{:.4f} | err5:{:.4f}".format(val_loss, err1, err5))

            fgsm_err1, fgsm_err5, fgsm_val_loss = self.valid_fgsm(val_loder)
            logger.info("\nvalid_fgsm | val_loss:{:.4f} | err1:{:.4f} | err5:{:.4f}".format(fgsm_val_loss, fgsm_err1, fgsm_err5))

            #PGD
            pgd_err1, pgd_err5, pgd_val_loss = self.valid_pgd(val_loder)
            logger.info("\nvalid_pgd | val_loss:{:.4f} | err1:{:.4f} | err5:{:.4f}".format(pgd_val_loss, pgd_err1, pgd_err5))

            if err1 <= best_err1:
                best_err1 = save_checkpoint(self.model, self.optimizer, e, train_losses=train_loss, val_loss=err1, check_loss=best_err1,
                                            savepath=self.config.work_dir, m_name=self.config.model_save_name)
                logger.info('Current Best (top-1 error) updated at epoch{}:{}'.format(e, best_err1))

            if err5 <= best_err5:
                best_err5 = err5
                logger.info('Current Best (top-5 error) updated at epoch{}:{}'.format(e, best_err5))

            if fgsm_err1 <= best_fgsmerr1:
                best_fgsmerr1 = save_checkpoint(self.model, self.optimizer, e, train_losses=train_loss, val_loss=fgsm_err1, check_loss=best_fgsmerr1,
                                            savepath=self.config.work_dir, m_name=self.config.model_save_name+"fgsm")
                logger.info('Current Best (FGSM top-1 error) updated at epoch{}:{}'.format(e, best_fgsmerr1))


            if pgd_err1 <= best_pgderr1:
                best_pgderr1 = save_checkpoint(self.model, self.optimizer, e, train_losses=train_loss, val_loss=pgd_err1, check_loss=best_pgderr1,
                                            savepath=self.config.work_dir, m_name=self.config.model_save_name+"pgd")
                logger.info('Current Best (PGD top-1 error) updated at epoch{}:{}'.format(e, best_pgderr1))

            if err1 < top_score[4][2]:
                top_score[4]=[e,val_loss,err1]
                z = np.argsort(top_score[:, 2])
                top_score = top_score[z]
                #best_err1 = save_checkpoint(self.model, self.optimizer, e, val_loss=err1, check_loss=best_err1,
                #                            savepath=self.config.work_dir, m_name=self.config.model_save_name)
            if err5 < top_score5[4]:
                top_score5[4]=err5
                z = np.argsort(top_score5)
                top_score5 = top_score5[z]

            if(data_config.tensorboard):
                writer.add_scalar('training loss', train_loss, e)
                writer.add_scalar('valing loss', val_loss, e)
                writer.add_scalar('err1', err1, e)
                writer.add_scalar('err5', err5, e)

        writer.close()
        logger.info('\nbest score:{}'.format(data_config.model_save_name))
        for i in range(5):
            logger.info(top_score[i])
        logger.info(top_score5,top_score[:, 0])
        logger.info('Best(top-1 and 5 error):',top_score[:, 1].mean(), best_err1, best_err5)

        logger.info("best accuracy:\n avg_acc1:{:.4f} | best_acc1:{:.4f} | avg_acc5:{:.4f} | | best_acc5:{:.4f} ".
              format(100 - top_score[:, 2].mean(), 100 - best_err1, 100 - top_score5.mean(), 100 - best_err5))



if data_config.load_from is not None:
    #model, optimizer, start_epoch = load_checkpoint(model, optimizer, data_config.load_from)
    model, _, _ = load_checkpoint(model, checkpoint_path = data_config.load_from)
    start_epoch = 0

else:
    start_epoch = 0

#***********- Showing Configs -*************
logger.info("Using Device:{}".format(device_name))
logger.info("Network:{}".format(model))
#summary(model, x.size())
logger.info("Dataset:{}".format(data_config.dataset))
logger.info("***********training_transforms:{}".format(data_config.training_transforms))
logger.info("***********validation_transforms:{}".format( data_config.validation_transforms))
logger.info("Mode:{}".format(data_config.config_mode))
if data_config.config_mode == "train":
    logger.info("***********Epoch:{}".format(data_config.epochs))
    logger.info("***********Batch_size:{}".format(data_config.batch_size))
    logger.info("***********rand_seed:{}".format( data_config.rand_seed))
    logger.info("***********optimizer:{}".format(data_config.optimizer))
    logger.info("***********optimizer_parm:{}".format(str(data_config.optimizer_parm)))
    logger.info("***********scheduler:{}".format(data_config.scheduler))
    logger.info("***********scheduler_parm:{}".format(str(data_config.scheduler_parm)))
    logger.info("***********loss_f:{}".format(data_config.loss_f))



Trainer = trainer(loss_f,loss_dv,model,optimizer,scheduler,config=data_config)
train = DataLoader(train_dataset, batch_size=data_config.batch_size, shuffle=True, num_workers=data_config.WORKERS, pin_memory=False)
val = DataLoader(val_dataset, batch_size=data_config.batch_size, shuffle=False, num_workers=data_config.WORKERS, pin_memory=False)

val_dataset_pgd = torch.utils.data.Subset(val_dataset, range(0, len(val_dataset), 10))
val_pgd = DataLoader(val_dataset_pgd, batch_size=data_config.batch_size, shuffle=False, num_workers=data_config.WORKERS, pin_memory=False)


if data_config.config_mode == "train":
    Trainer.run(train, val, data_config.load_from, start=start_epoch)
    logger.info('Finish training!')
else:
    if data_config.attack == 'none':
        Trainer.valid_epoch(val)
    elif data_config.attack == 'fgsm':
        Trainer.valid_fgsm(val)
    elif data_config.attack == 'pgd':
        Trainer.valid_pgd(val)

    logger.info('Finish validation!')


