
import math
import os
import torch
from tqdm import tqdm
from config import config
from functools import partial
from torch.utils.data import DataLoader

def epoch_train(train_loader, val_loader, model, optimizer, loss_fn, scaler, scaled_anchors, cur_epoch, all_epoch, loss_history,conf_thresh=0):
    # 设置一个进度条，便于我们查看训练的精度
    pmgressbar_train = tqdm(train_loader, desc=f"Train epoch {cur_epoch + 1}/{all_epoch}", postfix=dict, mininterval=0.3)
    # 防止外面让模型编程评价模式
    model.train()
    # 保存训练损失
    train_losses = []
    for iteration, (images, targets) in enumerate(train_loader):
        # 将数据加载到gpu或则cpu上面
        images = images.to(config.DEVICE)
        y0, y1, y2 = (
            targets[0].to(config.DEVICE),   
            targets[1].to(config.DEVICE),
            targets[2].to(config.DEVICE),
        )

        # 这里默认了使用gpu训练，
        with torch.cuda.amp.autocast():
            out = model(images)
            loss = (
                  loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )
        
        train_losses.append(loss.item())   
        optimizer.zero_grad()

        # 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 更新进度条
        train_mean_loss = sum(train_losses) / len(train_losses)
        pmgressbar_train.set_postfix(**{'train_loss' : train_mean_loss,
                                        'lr'         : get_lr(optimizer)})
        pmgressbar_train.update()

    pmgressbar_train.close()
    print("一个epoch的训练集的训练结束. ")
    print("开始验证集的测试 .")
    pmgressbar_val = tqdm(val_loader, desc=f"Train epoch {cur_epoch + 1}/{all_epoch}", postfix=dict, mininterval=0.3)

    # 这是使用训练集进行评价
    model.eval()
    val_losses = []
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0
    for iteration, (images, targets) in enumerate(val_loader):
        images = images.to(config.DEVICE)
        y0, y1, y2 = (
            targets[0].to(config.DEVICE),   
            targets[1].to(config.DEVICE),
            targets[2].to(config.DEVICE),
        )
        with torch.no_grad():
            out = model(images)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )
        val_losses.append(loss.item())
        val_mean_loss = sum(val_losses) / len(val_losses)
        pmgressbar_val.set_postfix(val_loss=val_mean_loss)
        pmgressbar_val.update()

    pmgressbar_val.close()
    print("一个epoch的验证集的验证结束. ")

    if conf_thresh != 0:
        print("计算每个类别细分的各个准确度. ")

        for i in range(3):
            targets[i] = targets[i].to(config.DEVICE)
            obj = targets[i][..., 0] == 1 # in paper this is Iobj_i
            noobj = targets[i][..., 0] == 0  # in paper this is Iobj_i

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == targets[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > conf_thresh
            correct_obj += torch.sum(obj_preds[obj] == targets[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == targets[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

        print(f"Class accuracy is: {(correct_class/(tot_class_preds+1e-16))*100:2f}%")
        print(f"No obj accuracy is: {(correct_noobj/(tot_noobj+1e-16))*100:2f}%")
        print(f"Obj accuracy is: {(correct_obj/(tot_obj+1e-16))*100:2f}%")

    # 将训练的损失画出来
    loss_history.append_loss(cur_epoch, train_mean_loss, val_mean_loss)

    # 保存网络的权重
    if config.SAVE_MODEL:
        if (cur_epoch + 1) % config.WEIGHT_SAVE_PERIOD == 0 or cur_epoch + 1 == config.NUM_EPOCHS:
            filename = os.path.join(config.SAVE_DIR, "checkpoint/ep%03d-train_loss%.3f-val_loss%.3f.pth"% (cur_epoch, train_mean_loss, val_mean_loss))
            save_checkpoint(model=model, optimizer=optimizer, filename=filename)

        elif len(loss_history.val_loss) <= 1 or (val_mean_loss) <= min(loss_history.val_loss):
            print('Save current best model to best_epoch_weights.pth')
            filename = "best_epoch_weights.pth"
            save_checkpoint(model=model, optimizer=optimizer, filename=filename)
        
        else: # 不然就是最后一个epoch了，保存最后一个epoch
            filename = "last_epoch_weights.pth"
            save_checkpoint(model=model, optimizer=optimizer, filename=filename)


#---------------------------------------------------#
#   加载数据集相应的 txt 
#---------------------------------------------------#
def get_anno_lines(train_annotation_path, val_annotation_path):
    with open(os.path.join(train_annotation_path, "train.txt")) as f:
        train_lines = f.readlines()
    with open(os.path.join(val_annotation_path, "val.txt")) as f:
        val_lines   = f.readlines()
    
    return train_lines, val_lines

#---------------------------------------------------#
#   从优化器中获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   对优化器中设置新的学习率
#---------------------------------------------------#
def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#---------------------------------------------------#
#   选择不同的学习率下降公式
#---------------------------------------------------#
def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    
    # 余弦退火算法
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr
    # step 下降算法
    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    # 返回有关装载了相关参数的函数
    return func

#---------------------------------------------------------------------------------------------------#
#   下面为保存和加载模型的相关参数
#---------------------------------------------------------------------------------------------------#

#---------------------------------------------------#
#   只是单纯的保存模型的权重，一般为.pt或则 .pth
#---------------------------------------------------#
def save_model_weight(model, filename ="my_model.pth"):
    print("=> Saving model weight. ")
    torch.save(model.state_dict(), filename)

#---------------------------------------------------#
#   单纯的加载模型的权重,一般为.pt或则 .pth
#---------------------------------------------------#
def load_model_weight(model, filename ="my_model.pth"):
    print("=> loading model weight. ")
    model.load_state_dict(torch.load(filename))

#---------------------------------------------------#
#   保存模型的Checkpoint,一般为.pt.tar或则 .pth.tar
#---------------------------------------------------#
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint. ")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

#---------------------------------------------------#
#   加载模型的Checkpoint,一般为.pt.tar或则 .pth.tar
#---------------------------------------------------#
def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

#---------------------------------------------------#
#   Checkpoint中保存自己想要的任意参数,一般为.pt.tar或则 .pth.tar
#---------------------------------------------------#    
def save_checkpoint_params(model, filename, *params):
    print("=> Saving checkpoint with param: ", params)

    checkpoint = {"state_dict": model.state_dict()}
    for param in params:
        checkpoint[param] = param.state_dict()
    
    torch.save(checkpoint, filename)

#---------------------------------------------------#
#   加载自己想要的参数,一般为.pt.tar或则 .pth.tar
#---------------------------------------------------# 
def load_checkpoint_params(model, checkpoint_file, *params, device):
    print("=> Loading checkpoint with param: ", params)
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    for param in params:
        if checkpoint.has_key(param):
            if param == 'optimizer':
                if checkpoint.has_key('lr'):
                    param.load_state_dict(checkpoint['param'])
                    for param_group in param.param_groups:
                        param_group["lr"] = checkpoint["lr"]
                else:
                    print("模型没有保存学习率, 加载优化器会出错")
        else:
            print("输入的要加载的参数中 {} 不再其中, 没有被加载".format(param))




