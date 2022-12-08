import torch
import sys
from utils import utils

from tqdm import tqdm
import numpy as np

import umap
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
sns.set_theme()
sns.set_style("whitegrid")
sns.despine()



@torch.no_grad()
def extractFeatures(dataLoader, model, dict_clss, maxLength=-1):
	realFeatures = list()
	realLabels = list()

	for im, cls_im in tqdm(dataLoader, leave=False, desc='extractFeatures', ncols=50, bar_format="{desc}|{bar}|{percentage:3.0f}% {elapsed}"):

		im = im.float().cuda()

		# Image embedding into a semantic space
		_, feat = model(im)
		feat = model.base_model.last_linear(feat)

		# Accumulate sketch embedding
		realFeatures.append(feat.cpu())

		cls_numeric = torch.from_numpy(utils.numeric_classes(cls_im, dict_clss)).long().cpu()

		realLabels.append(cls_numeric)
	realFeatures = torch.cat(realFeatures, 0)
	realLabels = torch.cat(realLabels, 0)
	if maxLength > 0 and len(realFeatures > maxLength):
		randIdx = torch.randperm(len(realFeatures))[:maxLength]
		realFeatures = realFeatures[randIdx]
		realLabels = realLabels[randIdx]
	else:
		randIdx = torch.randperm(len(realFeatures))
		realFeatures = realFeatures[randIdx]
		realLabels = realLabels[randIdx]

	return realFeatures, realLabels

def selectSubSet(features, labels, nclass):
	if isinstance(nclass, int):
		allLabels = torch.unique(labels)
		randIdx = torch.randperm(len(allLabels))[:nclass]
		selectedFeatures = list()
		selectedLabels = list()
		for idx in randIdx:
			targetLabel = allLabels[idx]
			selectedFeatures.append(features[labels == targetLabel])
			selectedLabels.append(labels[labels == targetLabel])
		return torch.cat(selectedFeatures), torch.cat(selectedLabels)
	else:
		selectedFeatures = list()
		selectedLabels = list()
		for targetLabel in nclass:
			selectedFeatures.append(features[labels == targetLabel])
			selectedLabels.append(labels[labels == targetLabel])
		return torch.cat(selectedFeatures), torch.cat(selectedLabels)


def draw_scatters(features, labels, class_num, metainfo):
	cmap = np.array(sns.color_palette("Spectral", n_colors=class_num))

	fig, ax1 = plt.subplots(1, 1, figsize=(8.27, 8.27), dpi=384)


	ax1.scatter(features[:, 0], features[:, 1], c=cmap[labels], s=2)
	ax1.tick_params(which='both', direction='in', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False, grid_alpha=0.6, labelsize=9, color="#888888")
	ax1.grid(True, which="minor", axis="both", lw=0.3, c="#aaaaaa")
	ax1.grid(True, which="major", axis="both")
	ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
	ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
	ax1.set_axisbelow(True)


	ax1.spines['bottom'].set_color('#00000000')
	ax1.spines['top'].set_color('#00000000')
	ax1.spines['right'].set_color('#00000000')
	ax1.spines['left'].set_color('#00000000')

	plt.tight_layout()

	plt.savefig("{}.pdf".format(metainfo), transparent=True, bbox_inches="tight")

	plt.clf()
	plt.close()

	print(f"raw feature UMAP saved at {metainfo}.pdf")


def lowerbound(features, labels):
	uniqueLabel = torch.unique(labels)
	intraSimilarities = list()
	centers = list()
	for l in uniqueLabel:
		featureOfThisLabel = features[labels == l]
		center = torch.mean(featureOfThisLabel, 0)
		intraDistance = (featureOfThisLabel * center).sum(-1).mean()
		intraSimilarities.append(intraDistance)
		centers.append(center)
	intraSimilarities = torch.tensor(intraSimilarities).mean()
	centers = torch.stack(centers, 0)
	interSimilarities = (centers @ centers.T).triu(1)
	interSimilarities = interSimilarities[interSimilarities > 0]
	interSimilarities = interSimilarities.mean()
	return intraSimilarities, interSimilarities

def dimension_reduction(features, labels):
	features = features.cpu().numpy()
	labels = labels.cpu().numpy()

	print("UMAP start")
	fit = umap.UMAP(low_memory=False, random_state=42)
	features = fit.fit_transform(features)
	print("UMAP end")
	return fit, features, labels
