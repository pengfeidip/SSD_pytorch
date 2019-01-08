"""
Time: 2019-1-8 11:34:56
Author: pengfei
Email: pegnfeidip@qq.com
Descriptions: compute Recall and FP rate for detection result.
"""


import torch
import os
import sys
import csv

import numpy as np
from tqdm import tqdm


VOC_CLASSES = (  # always index 0
	'aeroplane', 'bicycle', 'bird', 'boat',
	'bottle', 'bus', 'car', 'cat', 'chair',
	'cow', 'diningtable', 'dog', 'horse',
	'motorbike', 'person', 'pottedplant',
	'sheep', 'sofa', 'train', 'tvmonitor')

def list_reshape():
	"""
	:return: preds:(list) detection; labels:(list) labels;  test_list:(list) test images' name
	"""

	result_path = "/root/voc/VOCdevkit/VOC2007/results/"
	preds = []

	#  把detection的结果整理一下
	for class_name in VOC_CLASSES:

		with open(os.path.join(result_path, "det_test_" + class_name + ".txt")) as detect_file:

			lines = detect_file.readline()
			while lines:
				temp = lines.split(' ')
				temp[1:] = [float(x) for x in temp[1:]]
				temp.append(VOC_CLASSES.index(class_name))
				preds.append(temp)
				lines = detect_file.readline()

	label_path = "/root/voc/VOCdevkit/VOC2007/VOC2007_test.csv"
	labels = []

	test_list = []
	with open(label_path)as csvfile:
		groundtruth = list(csv.reader(csvfile))
		for gt in groundtruth:
			test_list.append(gt[0][:6])  # 记录test images' name

			num_bboxes = int((len(gt) - 1) / 5)
			for i_bboxes in range(num_bboxes):
				coord_class_idx = [float(x) for x in gt[1 + i_bboxes * 5:6 + i_bboxes * 5]]  # 获取 coordinate 和 class index
				labels.append([gt[0][:6]] + coord_class_idx)  # add image name

	return preds, labels, test_list

def compute_iou(box1, box2):
	'''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
	Args:
	  box1: (tensor) bounding boxes, sized [N,4].
	  box2: (tensor) bounding boxes, sized [M,4].
	Return:
	  (tensor) iou, sized [N,M].
	'''

	N = box1.size(0)
	M = box2.size(0)

	lt = torch.max(
		box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
		box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
	)

	rb = torch.min(
		box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
		box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
	)

	wh = rb - lt  # [N,M,2]
	wh[wh < 0] = 0  # clip at 0
	inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

	area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
	area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
	area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
	area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

	iou = inter.float() / (area1 + area2 - inter + 1e-5)

	return iou

"""
preds[ [im_name, confidence, xmin, ymin, xmax, ymax, class_index], ...]
labels[ [im_name, xmin, ymin, xmax, ymax, class_index],  ...]
"""

def recall_FP(conf_threshold, iou_threshold=0.5, care_class=True):
	"""
	:param conf_threshold:(float) conf_threshold  to filter bboxes
	:param iou_threshold:(float:0.5)  iou_threshold for NMS
	:param care_class:  (bool:True) care class or not. if False only care location
	:return:
	"""
	print("conf threshold: {}  iou threshold: {}, care_class: {} \n".format(conf_threshold,
	                                                                     iou_threshold,
	                                                                     care_class))
	preds, labels, test_list = list_reshape()

	FP = 0.0
	TP = 0.0
	num_detection = 0

	for test_name in tqdm(test_list):

		# get predictions and labels with the same class.
		# and predictions should have a valid conf threshold
		preds_specified = [x for x in preds if x[0] == test_name and x[1] >= conf_threshold]
		labels_specified = [x for x in labels if x[0] == test_name]

		# if no valid predictions
		if not preds_specified:
			continue

		num_detection += len(preds_specified)

		# extract the coordinate for NMS
		coord_preds = [x[2:6] for x in preds_specified]
		coord_labels = [x[1:5] for x in labels_specified]

		iou = compute_iou(torch.tensor(coord_preds), torch.tensor(coord_labels))
		mask = iou > iou_threshold

		# 统计每一个图的TP 和 FP的数量
		if care_class:
			sub_tp = (mask.sum(dim=1) > 0).sum().item()

			row, col = np.where(mask.numpy())
			for idx in range(len(row)):
				if preds_specified[row[idx]][6] != labels_specified[col[idx]][5]:
					sub_tp -= 1

			TP += sub_tp  # 这些检测结果是本算法承认的
			FP += len(preds_specified) - sub_tp
			if FP < 0:
				print(FP)
				break
		else:
			sub_tp = (mask.sum(dim=1) > 0).sum().item()
			TP += sub_tp
			FP += len(preds_specified) - sub_tp

	recall = TP / len(labels)
	FP_rate = FP / num_detection
	print("\n recall:{}, FP rate:{}".format(recall, FP_rate))

if __name__ == "__main__":
	print("-----------start --------------\n")
	recall_FP(conf_threshold=0.6, care_class=False)
