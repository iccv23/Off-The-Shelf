import os
import random
import time
import numpy as np
import math
import sys

import os
if os.environ["PWD"].startswith("/ai/base/"):
	sys.path.append('/ai/base/G-FIR/src/')
else:
	sys.path.append('/home/user/data/codes/G-FIR/src/')

import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import apex

from models.senet import se_resnet50
from data.Sketchy import sketchy_extended
from data.TUBerlin import tuberlin_extended
from data.DomainNet import domainnet
from data.dataloaders import BaselineDataset, PairedContrastiveImageDataset
from data.sampler import BalancedSampler
from models.lower_bound.lbq import PQLayer
from models.lower_bound.codebookInit import uniform_hypersphere
from utils import utils
from utils.metrics import compute_retrieval_metrics
from utils.logger import AverageMeter

class Normalize(nn.Module):
	def forward(self, x):
		return F.normalize(x)



class PQLayer(nn.Module):
	def __init__(self, feat_dim, M, K, alpha=1):
		super().__init__()
		self.feat_dim, self.M, self.K, self.D = feat_dim, M, K, feat_dim//M
		self.alpha = alpha



		preGenerated = F.normalize(torch.tensor(uniform_hypersphere(self.D, self.K)).float()) / math.sqrt(self.M)

		codebook = torch.empty(self.M, self.K, self.D) * float("inf")
		for m in range(self.M):
			# random rotation
			gaus = torch.randn(self.D, self.D)
			svd = torch.linalg.svd(gaus)
			orth = svd[0] @ svd[2]

			preGeneratedRot = preGenerated @ orth

			randIdx = torch.randperm(self.K)
			codebook[m].copy_(preGeneratedRot[randIdx])

		self._C = nn.Parameter(codebook, requires_grad=False)

		# self._C = nn.Parameter(nn.init.xavier_uniform_(torch.empty(
		# 	(self.M, self.K, self.D))), requires_grad=True)

	# def codebookLoss(self):
	# 	# [m, k, k]
	# 	distance = ((self._C[:, :, None, ...] - self._C[:, None, ...]) ** 2).sum(-1)
	# 	mask = torch.eye(self.K, device=self._C.device, dtype=self._C.dtype).expand(self.M, self.K, self.K) * 1e20
	# 	# [M]
	# 	minDistance = (distance + mask).reshape(self.M, -1).min(-1)[0]
	# 	return -minDistance.log().sum()


	def forward(self, x, *_):
		# self._codebook_normalization()
		# print("x:\n", x, "\nshape[bxfeat_dim]=", x.shape, end='\n\n')
		# x:[bxd]=>[bxMxD]
		x_ = x.reshape(x.shape[0], self.M, self.D)


		# [b, M, 1]
		# x_2 = (x ** 2).sum(-1, keepdim=True)


		# print("x_:\n", x_, "\nshape[bxMxD]=", x_.shape, end='\n\n')
		# print("_C:\n", _C, "\nshape[MxKxD]=", _C.shape, end='\n\n')
		# x_:[bxMxD], _C:[MxKxD] => ips:[bxMxK]
		ips = torch.einsum('bmd,mkd->bmk', x_, self._C) # - 0.5 * x_2
		# print("ips:\n", ips, "\nshape[bxMxK]=", ips.shape, end='\n\n')
		if not self.training: # hard assignment
			# codes:[bxMxK]
			codes = ips.argmax(dim=-1)
			codes = F.one_hot(codes, num_classes=self.K).float()
		else: # soft assignment
			# codes:[bxMxK]
			# scale for different M
			codes = F.softmax(ips / self.alpha * self.M, dim=-1)

			hard = F.one_hot(ips.argmax(dim=-1), num_classes=self.K).float()
			# STE
			codes = (hard - codes).detach() + codes
			# print("codes:\n", codes, "\nshape[bxMxK]=", codes.shape, end='\n\n')
		# x_hat_:[bxMxD]
		# _C:[MxKxD], codes:[bxMxK] => x_hat_:[bxMxD]
		x_hat_ = torch.einsum('mkd,bmk->bmd', self._C, codes)
		# print("x_hat_:\n", x_hat_, "\nshape[bxMxD]=", x_hat_.shape, end='\n\n')
		# x_hat:[bxMxD]=>[bxfeat_dim]
		x_hat = x_hat_.view(x_hat_.shape[0], -1)

		return x_hat, codes

	def intra_normalization(self, x):
		return F.normalize(x.view(x.shape[0], self.M, self.D), dim=-1).view(x.shape[0], -1)


class Encoder(nn.Module):
	secondOrderPrototypes: torch.Tensor
	sampleCount: torch.Tensor
	randomMask: torch.Tensor
	def __init__(self, semantic_dim, train_classes, M, K, alpha, pretrained='imagenet'):

		super().__init__()

		self.base_model = se_resnet50(pretrained=pretrained)
		feat_dim = self.base_model.last_linear.in_features
		self.base_model.last_linear = nn.Sequential(
			nn.Linear(feat_dim, semantic_dim)
		)
		self.quantizer = PQLayer(semantic_dim, M, K, alpha)

		# self.attention_layer = nn.Identity() # NonLocalBlock(feat_dim)

		self.train_clases = train_classes
		self.feat_dim = semantic_dim

		self.use_fp = False
		self.register_buffer("secondOrderPrototypes", F.normalize(nn.init.xavier_uniform_(torch.empty(train_classes, semantic_dim))))
		self.register_buffer("sampleCount", torch.zeros((train_classes, )))

		# self.shadowNetwork = se_resnet50()
		# feat_dim = self.shadowNetwork.last_linear.in_features
		# self.shadowNetwork.last_linear = nn.Sequential(
		# 	nn.Linear(feat_dim, semantic_dim)
		# )

		self.register_buffer("randomMask", torch.tensor([1] * int(train_classes * 0.85) + [0] * (train_classes - int(train_classes * 0.85))).bool())
		self.randomMask.data.copy_(self.randomMask[torch.randperm(train_classes)])

		self.affinityMapper = nn.TransformerDecoder(nn.TransformerDecoderLayer(semantic_dim, 8, 8 * semantic_dim, activation=F.gelu, batch_first=True, norm_first=True), 3)

	def get_attention_score(self, features):
		layer = self.affinityMapper.layers[0]

		# [45, 1, D]
		x = features[:, None]
		x = x + layer._sa_block(layer.norm1(x), None, None)
		mem = self.secondOrderPrototypes.expand(len(features), *self.secondOrderPrototypes.shape)
		# [45, 1, 300]
		_, attenion_score = layer.multihead_attn(layer.norm2(x), mem, mem, need_weights=True)
		return attenion_score[:, 0]

	def generateMask(self, oneHot):
		# N, C = oneHot.shape
		# [N, C]
		selfMask = oneHot.clone().bool()
		# [N, C]
		randIdx = torch.stack([torch.randperm(self.train_clases) for _  in range(len(selfMask))])
		randomMask = self.randomMask[randIdx]
		# maskAmount = torch.randint(4, oneHot.shape[-1] - 4, (N, ))

		# randomMask = list()
		# for m in maskAmount:
		# 	randIdx = torch.randperm(C)
		# 	mask = torch.tensor([1] * m + [0] * (C - m), device=oneHot.device)[randIdx]
		# 	randomMask.append(mask)
		# randomMask = torch.stack(randomMask)
		return torch.logical_or(selfMask, randomMask)

	def extractFromMapper(self, features, quantizeds, classes, temperature):
		n, m = classes.shape
		oneHot = F.one_hot(classes.reshape(-1), num_classes=self.train_clases).float()

		self.updateSecondOrderPrototypes(features, oneHot)


		# features = self.extract(self.shadowNetwork, images)

		# [N, C] bool mask, True-mask, False-visible
		mask = self.generateMask(oneHot)


		# features = self.shadowNetwork.features(images)

		# out = self.shadowNetwork.avg_pool(features)
		# if self.shadowNetwork.dropout is not None:
		# 	out = self.shadowNetwork.dropout(out)

		# features = out.view(out.size(0), -1)

		# features = self.shadowNetwork.last_linear(features)


		# use prototype to represent feature
		# mask: [N, C], 1. self-mask, 2. random-mask
		# [N, 1, D] <- [N, C, D], [N, 1, D]
		mappedFeature = self.affinityMapper(features[:, None], self.secondOrderPrototypes.expand(len(features), *self.secondOrderPrototypes.shape), memory_key_padding_mask=mask)[:, 0]

		mappedQuantizeds = self.quantizer(mappedFeature, temperature)[0].reshape(n, m, -1)
		mappedFeature = F.normalize(mappedFeature).reshape(n, m, -1)

		features = features.reshape(n, m, -1)
		quantizeds = quantizeds.reshape(n, m, -1)

		features = torch.cat([mappedFeature, features], 1)
		quantizeds = torch.cat([mappedQuantizeds, quantizeds], 1)

		return features, quantizeds

	@torch.no_grad()
	def updateSecondOrderPrototypes(self, quantizeds, oneHot):
		# [N, C, D] -> [C, D]
		average = (oneHot[..., None] * quantizeds[:, None]).mean(0)
		# [C]
		count = oneHot.sum(0)
		self.sampleCount.add_(count).clamp_max_(len(oneHot) / self.train_clases * 20)
		momentum = self.sampleCount / (self.sampleCount + count + 1e-8)
		decay = count / (self.sampleCount + count + 1e-8)
		self.secondOrderPrototypes.data.copy_(F.normalize(self.secondOrderPrototypes * momentum[:, None] + average * decay[:, None], dim=-1))

	@staticmethod
	def extract(se_resnet, img):
		features = se_resnet.features(img)
		# feat_attn = self.attention_layer(features)

		out = se_resnet.avg_pool(features)
		if se_resnet.dropout is not None:
			out = se_resnet.dropout(out)

		feat_final = out.view(out.size(0), -1)

		return se_resnet.last_linear(feat_final)


	def encode(self, img, *_):

		features = self.extract(self.base_model, img)

		# [B, C, D], [B, 1, D] -> [B, 1, D]
		mappedFeature = self.affinityMapper(features[:, None], self.secondOrderPrototypes.expand(len(features), *self.secondOrderPrototypes.shape))

		# [B, D]
		mappedFeature = mappedFeature[:, 0]

		if self.use_fp:
			x = F.normalize(mappedFeature)
			q, c = self.quantizer(mappedFeature, -1)
			return x, x, c

		return F.normalize(mappedFeature), *self.quantizer(mappedFeature, -1)


	def forward(self, img, temperature):
		features = self.extract(self.base_model, img)
		# x = self.quantizer.intra_normalization(x)
		return F.normalize(features), *self.quantizer(features, temperature)

	def change_precision(self, type):
		self.use_fp = type == "fp"
		return self


def crossAlignedContrastiveLoss(features, quantizeds, temperature=0.07, contrast_mode='all', base_temperature=0.07, labels=None, mask=None):
	"""
	Args:
		features: hidden vector of shape [bsz, n_views, ...].
		labels: ground truth of shape [bsz].
		mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
			has the same class as sample i. Can be asymmetric.
	Returns:
		A loss scalar.
	"""
	device = features.device

	if len(features.shape) < 3:
		raise ValueError('`features` needs to be [bsz, n_views, ...],'
							'at least 3 dimensions are required')
	if len(features.shape) > 3:
		features = features.view(features.shape[0], features.shape[1], -1)
		quantizeds = quantizeds.view(features.shape[0], features.shape[1], -1)

	batch_size = features.shape[0]
	if labels is not None and mask is not None:
		raise ValueError('Cannot define both `labels` and `mask`')
	elif labels is None and mask is None:
		mask = torch.eye(batch_size, dtype=torch.float32).to(device)
	elif labels is not None:
		labels = labels.contiguous().view(-1, 1)
		if labels.shape[0] != batch_size:
			raise ValueError('Num of labels does not match num of features')
		mask = torch.eq(labels, labels.T).float().to(device)
	else:
		mask = mask.float().to(device)

	contrast_count = features.shape[1]
	# [N, M, D] -> [M * N, D]
	contrast_feature = features.permute(1, 0, 2).reshape(batch_size * contrast_count, -1)
	contrast_q = quantizeds.permute(1, 0, 2).reshape(batch_size * contrast_count, -1)
	if contrast_mode == 'one':
		anchor_feature = quantizeds[:, 0]
		anchor_count = 1
	elif contrast_mode == 'all':
		anchor_feature = contrast_q
		anchor_count = contrast_count
	else:
		raise ValueError('Unknown mode: {}'.format(contrast_mode))

	# compute logits
	anchor_dot_contrast = torch.div(
		torch.matmul(anchor_feature, contrast_feature.T),
		temperature)
	# for numerical stability
	logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
	logits = anchor_dot_contrast - logits_max.detach()

	# tile mask
	mask = mask.repeat(anchor_count, contrast_count)
	# mask-out self-contrast cases
	logits_mask = torch.scatter(
		torch.ones_like(mask),
		1,
		torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
		0
	)
	mask = mask * logits_mask

	# compute log_prob
	exp_logits = torch.exp(logits) * logits_mask
	log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

	# compute mean of log-likelihood over positive
	mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

	# loss
	loss = - (temperature / base_temperature) * mean_log_prob_pos
	loss = loss.view(anchor_count, batch_size).mean()

	return loss


class WarmUpAndCosineDecayScheduler:
    def __init__(self, optimizer, start_lr, base_lr, final_lr,
                 epoch_num, warmup_epoch_num):
        self.optimizer = optimizer
        self.step_counter = 0
        decay_epoch_num = epoch_num - warmup_epoch_num
        warmup_lr_schedule = np.linspace(start_lr, base_lr, warmup_epoch_num)
        cosine_lr_schedule = final_lr + 0.5 * \
            (base_lr - final_lr) * (1 + np.cos(np.pi *
                                               np.arange(decay_epoch_num) / decay_epoch_num))
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    # step at each mini-batch
    def step(self):
        curr_lr = self.lr_schedule[self.step_counter]
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = curr_lr
        self.step_counter += 1
        return curr_lr


class Trainer:

	def __init__(self, args):

		self.step = 0

		self.args = args

		print('\nLoading data...')

		if args.dataset=='Sketchy':
			data_input = sketchy_extended.create_trvalte_splits(args)

		if args.dataset=='DomainNet':
			data_input = domainnet.create_trvalte_splits(args)

		if args.dataset=='TUBerlin':
			data_input = tuberlin_extended.create_trvalte_splits(args)

		if args.dataset=='DomainNet':
			save_folder_name = 'seen-'+args.seen_domain+'_unseen-'+args.holdout_domain+'_x_'+args.gallery_domain
			if not args.include_auxillary_domains:
				save_folder_name += '_noaux'
			self.args.epochs //= 4
			# self.args.early_stop //= 4
		elif args.dataset=='Sketchy':
			if args.is_eccv_split:
				save_folder_name = 'eccv_split'
			else:
				save_folder_name = 'random_split'
			self.args.epochs //= 2
			# self.args.early_stop //= 2
		else:
			save_folder_name = ''

		self.tr_classes = data_input['tr_classes']
		self.va_classes = data_input['va_classes']
		semantic_vec = data_input['semantic_vec']
		data_splits = data_input['splits']

		np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		use_gpu = torch.cuda.is_available()

		if use_gpu:
			cudnn.benchmark = True
			torch.cuda.manual_seed_all(args.seed)

		# Imagenet standards
		im_mean = [0.485, 0.456, 0.406]
		im_std = [0.229, 0.224, 0.225]

		# Image transformations
		image_transforms = {
			'train':
			transforms.Compose([
				transforms.RandomResizedCrop(args.image_size),
				transforms.RandomHorizontalFlip(),
				transforms.autoaugment.TrivialAugmentWide(interpolation=transforms.InterpolationMode.BILINEAR),
				transforms.ToTensor(),
				transforms.Normalize(im_mean, im_std),
				transforms.RandomErasing(inplace=True),
			]),

			'eval':
			transforms.Compose([
				transforms.Resize((args.image_size, args.image_size)),
				transforms.ToTensor(),
				transforms.Normalize(im_mean, im_std)
			]),
		}

		# class dictionary
		self.dict_clss = utils.create_dict_texts(set(data_input['tr_classes'] + data_input['va_classes'] + data_input['te_classes']))


		self.train_dict_clss = utils.create_dict_texts(data_input['tr_classes'])

		fls_tr = data_splits['tr']
		cls_tr = np.array([f.split('/')[-2] for f in fls_tr])
		dom_tr = np.array([f.split('/')[-3] for f in fls_tr])
		tr_domains_unique = np.unique(dom_tr)

		# doamin dictionary
		self.dict_doms = utils.create_dict_texts(tr_domains_unique)
		print(self.dict_doms)
		domain_ids = utils.numeric_classes(dom_tr, self.dict_doms)

		data_train = PairedContrastiveImageDataset(fls_tr, cls_tr, dom_tr, self.dict_doms, self.train_dict_clss, image_transforms['train'], 2, 3)
		train_sampler = BalancedSampler(domain_ids, args.batch_size//len(tr_domains_unique), domains_per_batch=len(tr_domains_unique))
		self.train_loader = DataLoader(dataset=data_train, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)

		data_va_query = BaselineDataset(data_splits['query_te'], transforms=image_transforms['eval'])
		data_va_gallery = BaselineDataset(data_splits['gallery_te'], transforms=image_transforms['eval'])

		# PyTorch valid loader for query
		self.va_loader_query = DataLoader(dataset=data_va_query, batch_size=args.batch_size*5, shuffle=False, num_workers=args.num_workers, pin_memory=True)
		# PyTorch valid loader for gallery
		self.va_loader_gallery = DataLoader(dataset=data_va_gallery, batch_size=args.batch_size*5, shuffle=False, num_workers=args.num_workers, pin_memory=True)


		self.G_va_loader_query = DataLoader(dataset=BaselineDataset(data_splits['query_va'], transforms=image_transforms['eval']), batch_size=args.batch_size*5, shuffle=False, num_workers=args.num_workers, pin_memory=True)
		self.G_va_loader_gallery = DataLoader(dataset=BaselineDataset(data_splits['gallery_va'], transforms=image_transforms['eval']), batch_size=args.batch_size*5, shuffle=False, num_workers=args.num_workers, pin_memory=True)

		print(f'#Tr samples:{len(data_train)}; #Val queries:{len(data_va_query)}; #Val gallery samples:{len(data_va_gallery)}.\n')
		print('Loading Done\n')

		# Model
		self.model = Encoder(args.feat_dim, len(self.tr_classes), args.num_codebooks, args.codebook_size, args.alpha, "imagenet").cuda()

		self.optimizer = apex.optimizers.FusedLAMB(self.model.parameters(), lr=args.lr, betas=(0.8, 0.85))

		self.lr_scheduler = WarmUpAndCosineDecayScheduler(optimizer=self.optimizer, start_lr=args.start_lr, base_lr=args.lr, final_lr=args.final_lr, epoch_num=args.epochs, warmup_epoch_num=args.warmup_epoch_num) if args.use_scheduler else None

		if args.dataset=='DomainNet' or (args.dataset=='Sketchy' and args.is_eccv_split):
			self.map_metric = 'aps@200'
			self.prec_metric = 'prec@200'
		else:
			self.map_metric = 'aps@all'
			self.prec_metric = 'prec@100'


		self.suffix = '_bits-'+str(args.num_codebooks * int(math.log2(args.codebook_size)))+'_m-'+str(args.num_codebooks)+'_k-'+str(args.codebook_size)+\
					  '_mode-'+str(args.mode)+'_hplambda-'+str(args.hp_lambda)+'_hpgamma-'+str(args.hp_gamma)+'_e-'+str(args.epochs)+'_es-'+str(args.early_stop)+'_opt-'+args.optimizer+\
					  '_bs-'+str(args.batch_size)+'_lr-'+str(args.lr)+'_l2-'+str(args.l2_reg)+\
					  '_seed-'+str(args.seed)+'_tv-'+str(True)

		# exit(0)
		path_log = os.path.join('./logs', args.dataset, save_folder_name, self.suffix)
		self.path_cp = path_log
		# Logger
		print('Setting logger...', end='')
		self.logger = SummaryWriter(path_log)
		print('Done\n')

		self.start_epoch = 0
		self.best_map = 0
		self.early_stop_counter = 0
		self.last_chkpt_name='init'
		self.batch_size = args.batch_size

		self.resume_from_checkpoint(args.resume_dict)

		self.temperature = args.alpha
		self.initTemperature = args.alpha
		self.endTemperature = self.initTemperature * 0.01

	def resume_from_checkpoint(self, resume_dict):

		if resume_dict is not None:
			print('==> Resuming from checkpoint: ',resume_dict)
			checkpoint = torch.load(os.path.join(self.path_cp, resume_dict+'.pth'))
			self.start_epoch = checkpoint['epoch']+1
			self.model.load_state_dict(checkpoint['model_state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			self.best_map = checkpoint['best_map']
			self.last_chkpt_name = resume_dict



	def do_epoch(self):
		self.temperature = (math.cos(self.current_epoch * 2 / self.args.epochs) + 1) / 2 * (self.initTemperature - self.endTemperature) + self.endTemperature
		self.logger.add_scalar("train/T", self.temperature, self.step)

		self.model.train()

		batch_time = AverageMeter()
		total_loss = AverageMeter()

		# Start counting time
		time_start = time.time()

		for i, (images, classes) in enumerate(self.train_loader):
			self.step += 1

			classes = classes.cuda(non_blocking=True)

			# Transfer im to cuda
			# [N, 2*nviews, c, h, w]
			images = images.cuda(non_blocking=True)
			n, m, c, h, w = images.shape
			images = images.reshape(n * m, c, h, w)

			self.optimizer.zero_grad()

			# forward data and produce features and codes for 2 views
			features, quantizeds, logits = self.model(images, self.temperature)

			# con = self.model.trainMapper(features, quantizeds, classes[:, None].expand(n, m))

			# if self.current_epoch < 2:
			# features = features.reshape(n, m, -1)
			# quantizeds = quantizeds.reshape(n, m, -1)


			# 	# features = torch.cat([features, quantizeds], 1)
			# 	# Optimize parameters
			features, quantizeds = self.model.extractFromMapper(features, quantizeds, classes[:, None].expand(n, m), self.temperature)

			con = crossAlignedContrastiveLoss(features, quantizeds, labels=classes)

			# [1, k]
			# logits = logits.mean((0, 1))[None, ...]
			# entropy = F.cross_entropy(logits, torch.ones_like(logits) / self.model.quantizer.K)

			# quantizeds = quantizeds.reshape(n, m, -1)
			# reg = F.mse_loss((quantizeds ** 2).sum(-1), torch.ones(n, m, device=quantizeds.device))

			# entropy = self.model.quantizer.codebookLoss()

			loss = con # + reg + 1e-3 * entropy
			loss.backward()

			self.logger.add_scalar("train/con", con, self.step)
			# self.logger.add_scalar("train/reg", reg, self.step)
			# self.logger.add_scalar("train/unc", uncertainty, self.step)
			# self.logger.add_scalar("train/ent", entropy, self.step)
			# self.logger.add_scalar("train/mapping", mappingLoss, self.step)

			self.optimizer.step()

			# Store losses for visualization
			total_loss.update(loss.item(), len(images))

			# time
			time_end = time.time()
			batch_time.update(time_end - time_start)
			time_start = time_end

			if (i + 1) % self.args.log_interval == 0:
				# for i, l in enumerate(codes[:, range(self.args.num_codebooks)]):
				# 	self.logger.add_histogram(f"train/codes{i}", l, self.step)
				print('[Train] Epoch: [{0}/{1}][{2}/{3}]\t'
					  # 'lr:{3:.6f}\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'net {net.val:.4f} ({net.avg:.4f})\t'
					  .format(self.current_epoch+1, self.args.epochs, i+1, len(self.train_loader), batch_time=batch_time, net=total_loss))

			# if (i+1)==50:
			#     break
		if self.lr_scheduler is not None:

			def step():
				curr_lr = self.optimizer.param_groups[0]["lr"]
				for param_group in self.optimizer.param_groups:
					param_group["lr"] = curr_lr * 0.8
				return curr_lr
			# curr_lr = self.lr_scheduler.step()
			self.logger.add_scalar('train/lr', step(), self.step)
		return {'net':total_loss.avg}


	def do_training(self):

		print('***Train***')

		for self.current_epoch in range(self.start_epoch, self.args.epochs):
			start = time.time()

			loss = self.do_epoch()

			# evaluate on validation set, map_ since map is already there
			print('\n***Validation***')
			valid_data = evaluate(self.va_loader_query, self.va_loader_gallery, self.model,
								  self.dict_clss)

			map_ = float(valid_data[self.map_metric])
			prec = float(valid_data[self.prec_metric])


			G_valid_data = evaluate(self.G_va_loader_query, self.G_va_loader_gallery, self.model,
								  self.dict_clss)

			G_map_ = float(G_valid_data[self.map_metric])
			G_prec = float(G_valid_data[self.prec_metric])

			end = time.time()
			elapsed = end-start

			print(f"Epoch Time:{elapsed//60:.0f}m{elapsed%60:.0f}s lr:{utils.get_lr(self.optimizer):.7f} mAP:{map_:.4f} prec:{prec:.4f}\n")

			if map_ > self.best_map:

				self.best_map = map_
				self.early_stop_counter = 0

				model_save_name = 'val_map-'+'{0:.4f}'.format(map_)+'_prec-'+'{0:.4f}'.format(prec)+'_Gmap-'+'{0:.4f}'.format(G_map_)+'_Gprec-'+'{0:.4f}'.format(G_prec)+'_ep-'+str(self.current_epoch+1)+self.suffix
				utils.save_checkpoint({
										'epoch':self.current_epoch+1,
										'model_state_dict':self.model.state_dict(),
										'optimizer_state_dict':self.optimizer.state_dict(),
										'best_map':self.best_map,
										'corr_prec':prec
										}, directory=self.path_cp, save_name=model_save_name, last_chkpt=self.last_chkpt_name)
				self.last_chkpt_name = model_save_name

			else:
				self.early_stop_counter += 1
				if self.args.early_stop==self.early_stop_counter:
					print(f"Validation Performance did not improve for {self.args.early_stop} epochs."
						  f"Early stopping by {self.args.epochs-self.current_epoch-1} epochs.")
					break

				print(f"Val mAP hasn't improved from {self.best_map:.4f} for {self.early_stop_counter} epoch(s)!\n")

			# Logger step
			self.logger.add_scalar('Val/map', map_, self.step)
			self.logger.add_scalar('Val/prec', prec, self.step)

		self.logger.close()

		print('\n***Training and Validation complete***')

@torch.no_grad()
def evaluate(loader_sketch, loader_image, model, dict_clss):

	# Switch to test mode
	model.eval()

	sketchEmbeddings = list()
	sketchLabels = list()

	for i, (sk, cls_sk) in enumerate(loader_sketch):

		sk = sk.float().cuda()

		# Sketch embedding into a semantic space
		x, q, c = model.encode(sk, -1)
		# Accumulate sketch embedding
		sketchEmbeddings.append(x)

		cls_numeric = torch.from_numpy(utils.numeric_classes(cls_sk, dict_clss)).long().cuda()

		sketchLabels.append(cls_numeric)


	sketchEmbeddings = torch.cat(sketchEmbeddings, 0)
	sketchLabels = torch.cat(sketchLabels, 0)

	realEmbeddings = list()
	realLabels = list()

	allCodes = list()

	for i, (im, cls_im) in enumerate(loader_image):

		im = im.float().cuda()

		# Image embedding into a semantic space
		x, q, c = model.encode(im, -1)

		allCodes.append(c)

		# Accumulate sketch embedding
		realEmbeddings.append(q)

		cls_numeric = torch.from_numpy(utils.numeric_classes(cls_im, dict_clss)).long().cuda()

		realLabels.append(cls_numeric)

	realEmbeddings = torch.cat(realEmbeddings, 0)
	realLabels = torch.cat(realLabels, 0)

	# [N, M, K]
	allCodes = torch.cat(allCodes)
	# [M, K]
	allCodes = allCodes.sum(0)

	# import matplotlib.pyplot as plt

	# for i, c in enumerate(allCodes[:, range(allCodes.shape[-1])]):
	# 	print(c.shape)
	# 	c /= c.sum()
	# 	print(c.max())
	# 	plt.bar(range(len(c)), c.cpu().numpy(), color="black", edgecolor="black")
	# 	plt.savefig(f"codes{i}.pdf")
	# 	plt.clf()
	# 	plt.close()

	print('\nQuery Emb Dim:{}; Gallery Emb Dim:{}'.format(sketchEmbeddings.shape, realEmbeddings.shape))
	eval_data = compute_retrieval_metrics(sketchEmbeddings, sketchLabels, realEmbeddings, realLabels)

	return eval_data

@torch.no_grad()
def visualize_result(loader_sketch, loader_image, model, dict_clss):
	# Switch to test mode
	model.eval()

	sketchEmbeddings = list()
	sketchLabels = list()
	sketchIdx = list()

	for i, (sk, cls_sk, indices) in enumerate(loader_sketch):

		sk = sk.float().cuda()

		# Sketch embedding into a semantic space
		x, q, c = model.encode(sk, -1)
		# Accumulate sketch embedding
		sketchEmbeddings.append(x)

		cls_numeric = torch.from_numpy(utils.numeric_classes(cls_sk, dict_clss)).long().cuda()

		sketchLabels.append(cls_numeric)
		sketchIdx.append(indices)


	sketchEmbeddings = torch.cat(sketchEmbeddings, 0)
	sketchLabels = torch.cat(sketchLabels, 0)
	sketchIdx = torch.cat(sketchIdx, 0)

	realEmbeddings = list()
	realLabels = list()
	realIdx = list()

	allCodes = list()

	for i, (im, cls_im, indices) in enumerate(loader_image):

		im = im.float().cuda()

		# Image embedding into a semantic space
		x, q, c = model.encode(im, -1)

		allCodes.append(c)

		# Accumulate sketch embedding
		realEmbeddings.append(q)

		cls_numeric = torch.from_numpy(utils.numeric_classes(cls_im, dict_clss)).long().cuda()

		realLabels.append(cls_numeric)
		realIdx.append(indices)

	realEmbeddings = torch.cat(realEmbeddings, 0)
	realLabels = torch.cat(realLabels, 0)
	realIdx = torch.cat(realIdx, 0)

	# [N, M, K]
	allCodes = torch.cat(allCodes)
	# [M, K]
	allCodes = allCodes.sum(0)

	print('\nQuery Emb Dim:{}; Gallery Emb Dim:{}'.format(sketchEmbeddings.shape, realEmbeddings.shape))
	eval_data = compute_retrieval_metrics(sketchEmbeddings, sketchLabels, realEmbeddings, realLabels)

	allIds = eval_data["allid"]
	allAP = eval_data["allAP"]

	argsort = torch.argsort(allAP)

	KK = 20
	highest10 = argsort[-KK:]
	lowest10 = argsort[:KK]
	highest10Id = allIds[highest10]
	lowest10Id = allIds[lowest10]

	result = {
		"highest": {},
		"lowest": {}
	}
	for j, ids in enumerate(highest10Id):
		result['highest'][int(sketchIdx[highest10[j]])] = [loader_sketch.dataset.get_original(int(sketchIdx[highest10[j]]))]
		for i in range(KK):
			result['highest'][int(sketchIdx[highest10[j]])].append(loader_image.dataset.get_original(int(ids[i])))

	for j, ids in enumerate(lowest10Id):
		result['lowest'][int(sketchIdx[lowest10[j]])] = [loader_sketch.dataset.get_original(int(sketchIdx[lowest10[j]]))]
		for i in range(KK):
			result['lowest'][int(sketchIdx[lowest10[j]])].append(loader_image.dataset.get_original(int(ids[i])))
	return result
