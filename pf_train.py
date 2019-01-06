from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import datetime
from tensorboardX import SummaryWriter


def str2bool(v):
	return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
	description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=24, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=0, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

if torch.cuda.is_available():
	if args.cuda:
		torch.set_default_tensor_type('torch.cuda.FloatTensor')
	if not args.cuda:
		print("WARNING: It looks like you have a CUDA device, but aren't " +
		      "using CUDA.\nRun with --cuda for optimal training speed.")
		torch.set_default_tensor_type('torch.FloatTensor')
else:
	torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
	os.mkdir(args.save_folder)


def train():
	if args.dataset == 'COCO':
		if args.dataset_root == VOC_ROOT:
			if not os.path.exists(COCO_ROOT):
				parser.error('Must specify dataset_root if specifying dataset')
			print("WARNING: Using default COCO dataset_root because " +
			      "--dataset_root was not specified.")
			args.dataset_root = COCO_ROOT
		cfg = coco
		dataset = COCODetection(root=args.dataset_root,
		                        transform=SSDAugmentation(cfg['min_dim'],
		                                                  MEANS))
	elif args.dataset == 'VOC':
		if args.dataset_root == COCO_ROOT:
			parser.error('Must specify dataset if specifying dataset_root')
		cfg = voc
		dataset = VOCDetection(root=args.dataset_root,
		                       transform=SSDAugmentation(cfg['min_dim'],
		                                                 MEANS))

	if args.visdom:
		import visdom
		viz = visdom.Visdom()

	ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
	net = ssd_net

	if args.cuda:
		net = torch.nn.DataParallel(ssd_net)
		cudnn.benchmark = True

	if args.resume:
		print('Resuming training, loading {}...'.format(args.resume))
		ssd_net.load_weights(args.resume)
	else:
		vgg_weights = torch.load(args.save_folder + args.basenet)
		print('Loading base network...')
		ssd_net.vgg.load_state_dict(vgg_weights)

	if args.cuda:
		net = net.cuda()

	if not args.resume:
		print('Initializing weights...')
		# initialize newly added layers' weights with xavier method
		ssd_net.extras.apply(weights_init)
		ssd_net.loc.apply(weights_init)
		ssd_net.conf.apply(weights_init)

	optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
	                      weight_decay=args.weight_decay)
	criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
	                         False, args.cuda)

	now_time = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%Y-%m-%d-%H-%M')
	writer = SummaryWriter("runs/" + now_time[5:])



	print('Loading the dataset...')

	epoch_size = len(dataset) // args.batch_size
	print('Training SSD on:', dataset.name)
	print('Using the specified args:')
	print(args)



	data_loader = data.DataLoader(dataset, args.batch_size,
	                              num_workers=args.num_workers,
	                              shuffle=True, collate_fn=detection_collate,
	                              pin_memory=True)



	net.train()
	# loss counters
	loc_loss = 0
	conf_loss = 0

	loc_loss_epoch = 0.0
	conf_loss_epoch = 0.0
	total_loss_epoch = 0.0

	# create batch iterator
	step_index = 0
	iteration = 0
	max_epoches = cfg["max_iter"] / epoch_size
	for epoch in range(int(max_epoches+1)):
		t0 = time.time()
		for images, targets in data_loader:
			iteration += 1

			if iteration in cfg['lr_steps']:
				step_index += 1
				adjust_learning_rate(optimizer, args.gamma, step_index)

			if args.cuda:
				images = images.cuda()
				for i in range(len(targets)):
					targets[i] = targets[i].float().cuda()
			else:
				images = images
				targets = targets
			# forward

			out = net(images)

			# backprop
			optimizer.zero_grad()
			loss_l, loss_c = criterion(out, targets)
			loss = loss_l + loss_c
			loss.backward()
			optimizer.step()
			t1 = time.time()
			loc_loss += loss_l.item()
			conf_loss += loss_c.item()

			loc_loss_epoch += loss_l.item()
			conf_loss_epoch += loss_c.item()
			total_loss_epoch += loss.item()
			writer.add_scalar("loss/total_loss_iterations", loss.item(), iteration)

			if (iteration + 1) % epoch_size == 0:
				writer.add_scalar("loss/loc_loss", loc_loss_epoch / len(dataset), epoch)
				writer.add_scalar("loss/conf_loss", conf_loss_epoch / len(dataset), epoch)
				writer.add_scalar("loss/total_loss", total_loss_epoch / len(dataset), epoch)
				loc_loss_epoch = 0.0
				conf_loss_epoch = 0.0
				total_loss_epoch = 0.0

			if (iteration + 1) % 10 == 0:
				print('timer: %.4f sec.' % (t1 - t0))
				print('iter [' + repr(iteration + 1) + "/" + str(cfg["max_iter"]) + '] Loss: %.4f ||' % (loss.item()),
				      end=' ')

			if (iteration+1) % 5000 == 0:
				print('Saving state, iter:', iteration)
				torch.save(ssd_net.state_dict(), 'weights/ssd300_voc_pf_' +
				           repr(iteration+1) + '.pth')

			t0 = time.time()

		torch.save(ssd_net.state_dict(),
	            args.save_folder + '' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
	"""Sets the learning rate to the initial LR decayed by 10 at every
		specified step
	# Adapted from PyTorch Imagenet example:
	# https://github.com/pytorch/examples/blob/master/imagenet/main.py
	"""
	lr = args.lr * (gamma ** (step))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def xavier(param):
	init.xavier_uniform(param)


def weights_init(m):
	if isinstance(m, nn.Conv2d):
		xavier(m.weight.data)
		m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
	return viz.line(
		X=torch.zeros((1,)).cpu(),
		Y=torch.zeros((1, 3)).cpu(),
		opts=dict(
			xlabel=_xlabel,
			ylabel=_ylabel,
			title=_title,
			legend=_legend
		)
	)


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
	viz.line(
		X=torch.ones((1, 3)).cpu() * iteration,
		Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
		win=window1,
		update=update_type
	)
	# initialize epoch plot on first iteration
	if iteration == 0:
		viz.line(
			X=torch.zeros((1, 3)).cpu(),
			Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
			win=window2,
			update=True
		)


if __name__ == '__main__':
	train()
