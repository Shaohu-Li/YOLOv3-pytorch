
import os
from re import M
import torch
import datetime
import torch.optim as optim
from torch.utils.data import DataLoader

from model.loss import YoloLoss
from model.model import YOLOv3
from model.dataloader import Dataset_loader, yolo_dataset_collate, train_transforms, val_transforms
from config import config
from utils.utils_train import load_checkpoint, epoch_train,get_lr_scheduler, set_optimizer_lr,get_anno_lines
from utils.utils_map import check_class_accuracy,mean_average_precision
from utils.utils_box import get_evaluation_bboxes
from utils.callbacks import LossHistory


if __name__ == "__main__":

    if torch.cuda.is_available():
        print("在 GPU 上面训练. ")
    else:
        print("在 CPU 上面训练. ")
    


    # 加载自己的网络模型
    model = YOLOv3(config.model_config, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    print("模型加载完毕... ")
    # 加载优化器
    if config.OPTIMIZER_TYPE == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=config.INIT_LEARNING_RATE, betas=(config.MOMENTUM, 0.999), weight_decay=config.WEIGHT_DECAY
        )
    if config.OPTIMIZER_TYPE == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=config.INIT_LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY, nesterov=True
        )
    print("优化器加载完毕... ")


    # 加载损失函数
    loss_fn = YoloLoss()
    print("损失函数加载完毕... ")

    # 加载损失函数记录记录和画图器
    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir         = os.path.join(config.SAVE_DIR, "loss/loss_" + str(time_str))
    loss_history    = LossHistory(log_dir=log_dir, model=model, input_shape=config.IMAGE_SIZE)
    print("损失函数日志记载函数加载完毕... ")

    # 使用混合精度进行训练
    scaler = torch.cuda.amp.GradScaler()

    # 加载先验框 3 * 3 * 2
    scaled_anchors = (torch.tensor(config.ANCHORS)).to(config.DEVICE)
    print("先验框加载完毕... ")

    # 加载数据和验证集的迭代对象
    train_annotaion_lines, val_annotation_lines = get_anno_lines(train_annotation_path=config.TRAIN_LABEL_DIR, val_annotation_path=config.VAL_LABEL_DIR)
    
    train_dataset   = Dataset_loader(annotation_lines=train_annotaion_lines, input_shape=config.IMAGE_SIZE, anchors=config.ANCHORS, 
                                    transform=train_transforms, train = True, box_mode="midpoint")
    val_dataset     = Dataset_loader(annotation_lines=val_annotation_lines, input_shape=config.IMAGE_SIZE,  anchors=config.ANCHORS,
                                    transform=val_transforms, train = True, box_mode="midpoint")
    train_loader    = DataLoader(train_dataset, config.BATCH_SIZE, config.SHUFFLR, num_workers=config.NUM_WORKERS, 
                                pin_memory=config.PIN_MEMORY, drop_last=True)
    val_loader      = DataLoader(val_dataset, config.BATCH_SIZE, config.SHUFFLR, num_workers=config.NUM_WORKERS, 
                                pin_memory=config.PIN_MEMORY, drop_last=False)
    print("数据集迭代器加载完毕...")

    # 判断是否需要夹杂之前保存的模型和相应的参数
    checkpoint_file = os.path.join(config.SAVE_DIR, config.LOAD_WEIGHT_NAME)
    if config.LOAD_MODEL:
        load_checkpoint(
            config.SAVE_DIR, model, optimizer, config.LEARNING_RATE
        )

    print("进入 epochs, 开始训练... ")
                        
    for epoch in range( config.NUM_EPOCHS ):

        print("Epoch:" + str(epoch) + " / " + str(config.NUM_EPOCHS) + "->")
        # 主训练函数
        epoch_train(train_loader, val_loader, model, optimizer, loss_fn, scaler, scaled_anchors, epoch, config.NUM_EPOCHS, loss_history)
        if epoch > 0 and epoch % 10 == 0:
            #---------------------------------------#
            #   评估一些当前的 mAP
            #---------------------------------------#
            pred_boxes, true_boxes = get_evaluation_bboxes(
                val_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")
            model.train()
        #---------------------------------------#
        #   1、获得学习率下降的公式
        #   2、设置新的学习率
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(config.LEARNING_RATE_DECAY_TYPE, config.INIT_LEARNING_RATE, config.MIN_LEARNING_RATE, config.NUM_EPOCHS)
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
    