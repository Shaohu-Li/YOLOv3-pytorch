import os
import random
from utils_voc.voc_annotation import convert_annotation

def rename_imgs(path):
    count=0 # 用来顺序的给文件夹命名
    for file in os.listdir(path): #该文件夹下所有的文件（包括文件夹）
        Olddir=os.path.join(path,file)   #原来的文件路径
        if os.path.isdir(Olddir):   #如果是文件夹则跳过
            continue
        filename=os.path.splitext(file)[0]   #文件名
        filetype=os.path.splitext(file)[1]   #文件扩展名
        Newdir=os.path.join(path,str(count).zfill(6)+filetype)  #用字符串函数zfill 以0补全所需位数
        os.rename(Olddir,Newdir)#重命名
        count += 1 # 将图片的名称加 1

def dataset_divide_annotation(path, trainval_percent = 0.8, train_percent = 0.6, mode = 0):
    """
    参数：
        path: 输入的路径
        trainval_percent: 训练集和验证集在整个数据集所占的比例
        train_percent   : 训练集在训练集和验证集所占的比例
        mode            : 
                        == 0 -> 代表同时生成 Mian 和 SelfMain 里面的文件划分
                        == 1 -> 代表仅仅生成 Main 里面的文件划分
                        == 2 -> 代表 进行生成 SelfMain 里面的文件划分
    """
    #---------------------------------------------------------------------#
    # 训练集:0.48    -> 大约 12639
    # 测试集:0.32    -> 大约 8426
    # 验证集:0.2     -> 大约 5267
    #---------------------------------------------------------------------#
    
    txt_name_list = ["train", "val", "test"]

    if mode == 0 or mode == 1:
        # 确定要读写的xml文件夹和要保存到的文件
        xmlfilepath  = os.path.join(path, "\\Annotations")
        txtsavepath  = os.path.join(path, "\\ImageSets\\Main")

        temp_xml     = os.listdir(xmlfilepath)
        total_xml    = []
        # 为了防止 文件中存在其他的文件
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        num = len(total_xml)

        # 随机确定那些图片将被划分到训练集、验证集和测试集
        tv       = int(num * trainval_percent)
        tr       = int(tv * train_percent)
        trainval = random.sample(range(num), tv)
        train    = random.sample(trainval, tr)

        print("train and val size",tv)
        print("train size",tr)

        # 打开四个单纯写入划分图片名称的文件
        ftrainval       = open( os.path.join(txtsavepath, '\\trainval.txt'), 'w' )  
        ftest           = open( os.path.join(txtsavepath, '\\test.txt'), 'w' )
        ftrain          = open( os.path.join(txtsavepath, '\\train.txt'), 'w' )
        fval            = open( os.path.join(txtsavepath, '\\val.txt'), 'w' )


        for i in range(num):
            name = total_xml[i][:-4] + '\n'
            if i in trainval:
                ftrainval.write(name)  
                if i in train: # 不是 训练集 肯定就是 验证集
                    ftrain.write(name)
                else:
                    fval.write(name)
            else: # 不是 训练集 和 验证集，就是 测试集
                ftest.write(name)

        ftrainval.close()  
        ftrain.close()
        fval.close()
        ftest.close()
    
    if mode == 0 or mode == 2:
        print( "Generate Main/annotation_train.txt、Main/annotation_val.txt and Main/annotation_test.txt for train." )
        for txt_name in txt_name_list: 
            img_ids = open( os.path.join(path, "ImageSets\\Main\\%s.txt" % (txt_name)), encoding='utf-8' ).read().strip().split()
            self_txt_file = open( os.path.join(path, 'ImageSets\\Main\\annotation_%s.txt' % txt_name), 'w', encoding='utf-8' )
            for img_id in img_ids:
                # print(img_id)
                self_txt_file.write("%s/JPEGImages/%s.jpg" % ( os.path.abspath(path), img_id ))
                convert_annotation(img_id, path, self_txt_file)
                self_txt_file.write('\n')

if __name__ == "__main__":

    path = "自己的数据集根目录"
    dataset_divide_annotation(path)