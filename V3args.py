import os
from models.AttackNetwork.mnist_net import mnist_net
from models.AttackNetwork.image_net import image_net,image_net_pretrained,densenet121_pretrained, resnet50_pretrained
import time
import torchvision.transforms as transforms

from onnx2torch import convert

# Path to ONNX model
#onnx_model_path = '/home/yinjie/FYP_/torch/models/ONNX/TinyImageNet_resnet_medium.onnx'
# You can pass the path to the onnx model to convert it or...
#onnx_model = convert(onnx_model_path)

class data_config:
    """***********- Attack Arguments-*************"""
    config_mode = "train"#"train"  # "val"
    attack = "fgsm"   #choices=['pgd', 'fgsm', 'none']
    epsilon = 4/255
    attack_iters = 20
    alpha = 3/255 #3/255 #0.375
    restarts = 1

    '''***********- dataset and directory-*************'''
    dataset= 'tiny_imagenet'#'mnist'#'imagenet'
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

    input_size = 128
    num_class = 200
    data_path = '/hdd7/yinjie'#'./dataset'
    #train_file=''
    #val_file = ''
    #test_file = ''

    '''**************- Model Configs -************************'''
    model_arch = resnet50_pretrained()#resnet50_pretrained()#densenet121_pretrained()
    model_save_name = "resnet50_pretrained_eps4255"
    model_save_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    writer_log = model_save_name+"_"+model_save_time
    #具体到pkl文件,load_from只加载权重，resume_from加载optimizer和权重等信息
    load_from = None#'/home/yinjie/FYP_/torch/ckpts/resnet50_pretrained_none_tiny_imagenet_2022-11-11_19:02:29/val_acc_25.799999999999997_epoch_28.pkl'
    resume_from = None#'/home/yinjie/FYP_/torch/ckpts/resnet50_pretrained_none_tiny_imagenet_2022-11-11_19:02:29/val_acc_25.799999999999997_epoch_28.pkl'
    #model_attack_dataset_time
    MODEL_PATH = '/hdd7/yinjie/FYP_ckpt'#'./ckpts/'
    work_dir = os.path.join(MODEL_PATH,model_save_name+"_"+attack+"_"+dataset+"_"+model_save_time)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    #model_save_path = MODEL_PATH + model_save_time


    '''***********- Train Arguments-*************'''

    '''***********- Hyper Arguments-*************'''
    autoaug = 0  # Auto enhancement set to 1
    gpus=[0,1]  #[1,2,3]
    WORKERS = 1
    tensorboard= True
    epochs = 50
    batch_size = 256
    #delta = 0.00001
    rand_seed = 0   #Fixed seed greater than 0
    lr = 0.1
    warm = 1 #warm up training phase
    #optimizer = "torch.optim.SGD"
    #optimizer_parm = {'lr': lr,'momentum':0.9, 'weight_decay':5e-4, 'nesterov':False}
    optimizer = "torch.optim.AdamW"
    optimizer_parm = {'lr': lr, 'weight_decay': 0.00001}
    #学习率：小的学习率收敛慢，但能将loss值降到更低。当使用平方和误差作为成本函数时，随着数据量的增多，学习率应该被设置为相应更小的值。adam一般0.001，sgd0.1，batchsize增大，学习率一般也要增大根号n倍
    #weight_decay:通常1e-4——1e-5，值越大表示正则化越强。数据集大、复杂，模型简单，调小；数据集小模型越复杂，调大。
    #scheduler ="torch.optim.lr_scheduler.MultiStepLR"
    #scheduler_parm ={'milestones':[15,30,60,120], 'gamma':0.1}

    #scheduler = "torch.optim.lr_scheduler.CyclicLR"
    #scheduler_parm ={'base_lr': 5e-4, 'max_lr':lr,'cycle_momentum':False}
    scheduler = "torch.optim.lr_scheduler.CosineAnnealingLR"
    scheduler_parm = {'T_max': 5, 'eta_min': 1e-6}
    # scheduler = "torch.optim.lr_scheduler.StepLR"
    # scheduler_parm = {'step_size':1000,'gamma': 0.65}
    # scheduler = "torch.optim.lr_scheduler.ReduceLROnPlateau"
    # scheduler_parm = {'mode': 'min', 'factor': 0.8,'patience':10, 'verbose':True,'threshold':0.0001, 'threshold_mode':'rel', 'cooldown':2, 'min_lr':0, 'eps':1e-08}
    # scheduler = "torch.optim.lr_scheduler.ExponentialLR"
    # scheduler_parm = {'gamma': 0.1}
    loss_f ='torch.nn.CrossEntropyLoss'
    loss_dv = 'torch.nn.KLDivLoss'
    #loss_fn = 'torch.nn.BCELoss' # loss_fn = 'torch.nn.BCEWithLogitsLoss'  # loss_fn='torch.nn.MSELoss'
    #fn_weight =[3.734438666137167, 1.0, 1.0, 1.0, 3.5203138607843196, 3.664049338245769, 3.734438666137167, 3.6917943287286734, 1.0, 3.7058695139403963, 1.0, 2.193419513003608, 3.720083373160097, 3.6917943287286734, 3.734438666137167, 1.0, 2.6778551377707998]

