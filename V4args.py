import os
from models.AttackNetwork.mnist_net import mnist_net
from models.AttackNetwork.image_net import image_net,image_net_pretrained,densenet121_pretrained, resnet50_pretrained
import time
import torchvision.transforms as transforms
import torch



class data_config:

    """***********-  Attack Arguments-*************"""
    alpha = 2/255 # 10 / 255 #0.3
    epsilon = 8 / 255 #1e-2
    attack_iters = 40
    restarts = 10

    '''***********- Post Train Arguments-*************'''
    #data used for pt
    pt_data = "ori_neigh" #ori_neigh, ori_rand, train
    #attack on pt_data, 也就是对pt_data采用什么attack,attack之后就是对抗样本, adv-fgsm,dir_adv - pgd, normal - clean
    pt_method = "dir_adv"
    #当且仅当用pgd时有效，pt_method为dir_adv时生效; 梯度方向
    adv_dir = "pos" #pos, neg, both， na
    #用来给PGD得到neighbour class的方法，untargeted 和 targeted
    neigh_method = "untargeted"#targeted,untargeted
    # 一次post_train经历几次iter
    pt_iter = 50
    #post_train的学习率
    pt_lr = 0.001
    #attack_iters用于PGD攻击的iter数
    attack_iters = 40
    #attack_restart
    attack_restart = 1

    # check args validity
    if adv_dir != 'na':
        assert pt_method == 'dir_adv'
    if pt_method == 'dir_adv':
        assert adv_dir != 'na'







    '''***********- dataset and directory-*************'''
    gpus = [0,1]
    batch_size = 100
    dataset= "tiny_imagenet"#'tiny_imagenet'#'mnist'#'imagenet'
    #Image Normalize Configs
    normalized = True
    normalized_cfg_mean = [0.4802, 0.4481, 0.3975]
    normalized_cfg_std = [0.2302, 0.2265, 0.2262]
    training_transforms = transforms.Compose([
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          ])
    validation_transforms = transforms.Compose([
                            transforms.ToTensor(),
                            ])

    input_size = 64
    num_class = 200
    data_path = '/hdd7/yinjie'#'./dataset'

    '''**************- Model Configs -************************'''
    model_arch = resnet50_pretrained()#mnist_net()#densenet121_pretrained()
    model_save_name = "resnet50_pgde8a3_atkpgde8a2_ptpgdpos"
    model_save_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    writer_log = model_save_name+"_"+model_save_time
    #具体到文件,load_from_pkl加载经过自己代码得到的模型dict,load_from_pth加载别人的模型权重用load_state_dict
    load_from_pkl = "/hdd7/yinjie/FYP_ckpt/resnet50_pretrained_pgd_tiny_imagenet_2023-01-09_11:44:13/resnet50_pretrained_val_acc_30.28_epoch_16.pkl"#"/home/yinjie/FYP_/torch/ckpts/Resnet18_none_tiny_imagenet_2022-10-31_19:41:49/val_acc_31.430000000000007_epoch_24.pkl"#None
    load_from_pth = None#"/home/yinjie/FYP_/torch/ckpts/mnist/fgsm.pth"
    # check args validity
    if load_from_pkl is not None:
        assert load_from_pth is None
    if load_from_pth is not None:
        assert load_from_pkl is None

    #aaa = torch.load(load_from_pkl)
    #print(aaa['net_state_dict'])




    #model_attack_dataset_time
    MODEL_PATH = '/hdd7/yinjie/FYP_ckpt/pt_results/'
    work_dir = os.path.join(MODEL_PATH,model_save_name+"_"+dataset+"_"+model_save_time)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
