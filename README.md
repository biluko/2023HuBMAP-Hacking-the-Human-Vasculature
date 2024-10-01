# 2023HuBMAP-Hacking-the-Human-Vasculature
这是一个从人体组织结构中识别血管组织的实例分割竞赛，竞赛评价指标为mAP@0.6。

1.	验证集划分策略：比赛训练数据包含噪音数据与正确标注的数据，利用所有噪音数据训练，正确标注数据作为验证。
2.	训练策略：利用Cascade Mask R-CNN作为检测框架，分别利用多种预训练模型作为特征提取网络，进行微调训练，继承MMDetection的默认参数，修改ROI的损失函数，提高bbox识别精度，采用三分类目标函数，以血管识别精度的验证分数作为衡量保存最佳权重。
3.	推理策略：每个单模型都采用多种尺寸进行TTA推理，以NMS算法融合预测结果，在此基础将不同模型的预测结果继续使用NMS融合得到最终预测，此阶段的NMS的IOU选择阈值与bbox分数过滤阈值均使用验证集来获取最佳。

特征提取Backbone：
swin_transformer, convnext-xlarge, efficientnet-b8

检测框架：
来自MMDetection的Cascade Mask R-CNN
