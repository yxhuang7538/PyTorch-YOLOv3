from __future__ import division # 引入最新版本的除法机制

# 引入自己写的函数
from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.augmentations import *
from utils.transforms import *
from utils.parse_config import *
from utils.loss import compute_loss
from test import evaluate

# 引入额外安装库
from terminaltables import AsciiTable # 在命令行里显示表格

import os
import sys
import time
import datetime
import argparse # 命令行解析模块
import tqdm # 显示进度条

import torch
from torch.utils.data import DataLoader
from torchvision import datasets # 加载常见数据集
from torchvision import transforms # 图像变换
from torch.autograd import Variable
import torch.optim as optim # 优化算法库


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch-YOLOv3") # 命令行解析的入口点
    #=====================================================添加参数=========================================================================#
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--verbose", "-v", default=False, action='store_true', help="Makes the training more verbose") # store_true默认存储的是False值
    parser.add_argument("--logdir", type=str, default="logs", help="Defines the directory where the training log files are stored")
    opt = parser.parse_args() # 将输入的参数进行存储
    print(opt)

    logger = Logger(opt.logdir) # 训练log保存位置

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 是否使用GPU

    os.makedirs("output", exist_ok=True) # exist_ok 只有在目录不存在时创建目录，目录已存在时不会抛出异常
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config) # 读取data配置文件
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device) # 将模型copy到gpu上一份（如果gpu存在的话）
    model.apply(weights_init_normal) # 初始化模型的所有参数

    # If specified we start from checkpoint
    # 从之前训练接着训练，也就是断点训练
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            # 加载你自己训练的模型文件
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            # 加载官方的预训练模型（.weights文件）
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    # 加载数据
    # ListDataset 加载图像标签以及图像加强的数据
    dataset = ListDataset(train_path, multiscale=opt.multiscale_training, img_size=opt.img_size, transform=AUGMENTATION_TRANSFORMS)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size= model.hyperparams['batch'] // model.hyperparams['subdivisions'], # batch_size = batch // subdivisions，//取整除 - 返回商的整数部分
        shuffle=True, # 设置为True时，会在每个epoch重新打乱数据
        num_workers=opt.n_cpu, # 用多少个子进程加载数据。0表示数据将在主进程中加载（默认0）
        pin_memory=True, # 将数据放入固定内存中，从而更快的将数据传输到gpu中
        collate_fn=dataset.collate_fn, # 传入一组list数据，经过我们自定义的一些处理，再传出来
    )

    # 优化器默认选择adam，此处亦可选择sgd优化器，其余可以自己继续定义
    if (model.hyperparams['optimizer'] in [None, "adam"]):
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
            )
    elif (model.hyperparams['optimizer'] == "sgd"):
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
            momentum=model.hyperparams['momentum'])
    else:
        print("Unknown optimizer. Please choose between (adam, sgd).")

    # 开始训练模型
    for epoch in range(opt.epochs):
        print("\n---- Training Model ----")
        model.train()
        start_time = time.time()
        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标

        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
            batches_done = len(dataloader) * epoch + batch_i # 已经处理过的batch

            imgs = imgs.to(device, non_blocking=True) # 允许调用者在不需要时绕过同步
            targets = targets.to(device) # 将targets放到gpu上

            outputs = model(imgs) # 得到模型计算的输出结果

            loss, loss_components = compute_loss(outputs, targets, model) # 计算loss

            loss.backward() # 进行反传

            ###############
            # Run optimizer
            ###############

            if batches_done % model.hyperparams['subdivisions'] == 0: # 如果已进行的batch除以subdivisions余数为0，代表已经完整的进行了batch-size
                # Adapt learning rate
                # Get learning rate defined in cfg
                lr = model.hyperparams['learning_rate'] # 获取学习率
                if batches_done < model.hyperparams['burn_in']: # batches_done小于burn_in，则采用下面的更新学习率方式
                    # Burn in
                    lr *= (batches_done / model.hyperparams['burn_in'])
                else:
                    # Set and parse the learning rate to the steps defined in the cfg
                    # 否则按照指定的steps对学习率进行衰减
                    for threshold, value in model.hyperparams['lr_steps']:
                        if batches_done > threshold:
                            lr *= value
                # Log the learning rate
                # 记录学习率
                logger.scalar_summary("train/learning_rate", lr, batches_done)
                # Set learning rate
                for g in optimizer.param_groups:
                        g['lr'] = lr

                # Run optimizer
                optimizer.step()
                # Reset gradients
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------
            log_str = ""
            log_str += AsciiTable(
                [
                    ["Type", "Value"],
                    ["IoU loss", float(loss_components[0])],
                    ["Object loss", float(loss_components[1])], 
                    ["Class loss", float(loss_components[2])],
                    ["Loss", float(loss_components[3])],
                    ["Batch loss", to_cpu(loss).item()],
                ]).table

            if opt.verbose: print(log_str)

            # Tensorboard logging
            tensorboard_log = [
                    ("train/iou_loss", float(loss_components[0])),
                    ("train/obj_loss", float(loss_components[1])), 
                    ("train/class_loss", float(loss_components[2])),
                    ("train/loss", to_cpu(loss).item())]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            # 评估模型
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            metrics_output = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.1,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=model.hyperparams['batch'] // model.hyperparams['subdivisions'],
            )
            
            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                evaluation_metrics = [
                ("validation/precision", precision.mean()),
                ("validation/recall", recall.mean()),
                ("validation/mAP", AP.mean()),
                ("validation/f1", f1.mean()),
                ]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)

                if opt.verbose:
                    # Print class APs and mAP
                    ap_table = [["Index", "Class name", "AP"]]
                    for i, c in enumerate(ap_class):
                        ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
                    print(AsciiTable(ap_table).table)
                    print(f"---- mAP {AP.mean()}")                
            else:
                print( "---- mAP not measured (no detections found by model)")

        # 保存模型
        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
            model.save_darknet_weights("checkpoints/yolov3_ckpt_%d.weights" % epoch)
