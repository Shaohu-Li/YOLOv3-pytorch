import torch
from tqdm import tqdm
from config import config

from collections import Counter
from utils.utils import intersection_over_union

def check_class_accuracy(model, loader, threshold):
    """
    得到当前验证集的分类的精确度
    """
    model.eval()
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(config.DEVICE)
        with torch.no_grad():
            out = model(x)

        for i in range(3):
            y[i] = y[i].to(config.DEVICE)
            obj = y[i][..., 0] == 1 # in paper this is Iobj_i
            noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

    print(f"Class accuracy is: {(correct_class/(tot_class_preds+1e-16))*100:2f}%")
    print(f"No obj accuracy is: {(correct_noobj/(tot_noobj+1e-16))*100:2f}%")
    print(f"Obj accuracy is: {(correct_obj/(tot_obj+1e-16))*100:2f}%")
    model.train()

def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=80, ap_print = True
):
    """

    计算 mean average precision (mAP)

    Parameters:
        pred_boxes (list)    : 网络预测的所有的检测框(在 NMS 之后的),
                             : 单一检测框的形式为 [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
                             : train_idx, 指示图片编号, 用于区分不同的图片
        true_boxes (list)    : 真实的检测框标签
        iou_threshold (float): 和真实检测框的 iou 阈值
        box_format (str)     : 框的坐标形式
        num_classes (int)    : 当前数据集的类别总数
        ap_print             : 是否打印每一类的 AP 的值

    Returns:
        float: 返回所有类别总的 mAP 
    """
    #  AP学习链接 : https://blog.csdn.net/qq_36523492/article/details/108469465 

    #               P(正样本)      N(负样本)
    #   T(正确分类)   TP            TN
    #   F(错误分类)   FP            FN 
    # 
    #   Precision = TP / (TP + FP)  : 衡量网络输出的检测框中, 多少是正确的检测框
    #   Recall    = TP / (TP / FN)  : 衡量一张图片中的检测框中,多少被正确检测出来 (FN:实际上是没有正类分对,还是正类)
    #   Accuracy  = (TP + TN) / (TP + TN + FP + FN)
    #
    #   每一张图片都有 Precision Recall 吧, 在坐标上绘制所有点求平均估值, 就是 AP
    #   对所有类求 AP 的均值就是 MAP

    # list storing all AP for respective classes
    # 保存每一类的预测的准确度
    average_precisions = []

    # 用于数值稳定性
    epsilon = 1e-6

    for c in range(num_classes):
        # 分别用于保存当前类别的检测框和真值框
        detections = []
        ground_truths = []

        # 只计算当前 类别 的准确度
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # 我们要计算出当前类中的,每一张图片的 Precision和 Recall
        # 算出每一副图片上面检测框的数量,并形成字典,如{0:3, 1:5, 2.....}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # 转换 amount_bboxes 的形式
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # 按照网络输出检测框的分数从大到小进行排序
        detections.sort(key=lambda x: x[2], reverse=True)
        # TP 为 正确分类 正类, FP 为 错误分类 正类
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))

        # 我们需要分类正确总数 (TP / FN)
        total_true_bboxes = len(ground_truths)

        # 如果当前类的真实框数量为零的话,我们将忽略这个类别的计算
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            # 检查检测框的标签,在当前类包含的图片中找到对应的图片上的检测框
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            # 计算当前检测框和图片上所有检测框对应的 iou
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            # 大于阈值就认为我们为当前图片上的当前的真实检测框找到了对应的预测框
            if best_iou > iou_threshold:
                # 一个真实框只能对应检测框,后面的就算是错误分类了
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        # 计算当前类的总的 TP 和 FP
        # 如 TP = [1, 0, 1, 1, 0] -> [3]
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)

        # 把[0,1]这个点加入其中
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        # torch.trapz 是使用梯度积分的方式
        class_ap = torch.trapz(precisions, recalls)
        if ap_print:
            print("The current class {} AP : {}".format(c, class_ap))
        average_precisions.append(class_ap)

    return sum(average_precisions) / len(average_precisions)