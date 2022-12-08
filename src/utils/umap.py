import torch
import sys
from utils import utils

from tqdm import tqdm
import numpy as np



from scipy.spatial import Voronoi, voronoi_plot_2d

import umap
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib import rcParams
from matplotlib import rc
sns.set_theme()
sns.set_style("whitegrid")
sns.despine()

from adjustText import adjust_text


rcParams['font.family'] = 'Times New Roman'
rcParams["axes.labelweight"] = "bold"
rcParams["font.weight"] = "bold"
rcParams['mathtext.rm'] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.it'] = 'Times New Roman:italic'
rcParams['mathtext.bf'] = 'Times New Roman:bold'
# rc('text', usetex=True)


palette = ["#264656", "#2A9D8F", "#1F363D", "#457B9D", "#34A0A4", "#B5838D", "#22223B", "#9A8C98", "#9E0059", "#390099", "#B298DC", "#8E9AAF", "#DDA15E", "#EDFF71", "#2E6F95", "#F85E00", "#212F45", "#4D7F22", "#C99A42", "#EF6351", "#001F54", "#1282A2", "#CC5803", "#A5E6BA", "#041F1E", "#858AE3", "#EBCFB2", "#9DD9D2", "#FDCA40", "#92B4A7"]


def convert(hexSting):
	r, g, b = hexSting[1:3], hexSting[3:5], hexSting[5:]
	r, g, b = int(r, 16), int(g, 16), int(b, 16)
	return [r / 255.0, g / 255.0, b / 255.0]

palette = np.array([convert(i) for i in palette])

@torch.no_grad()
def extractFeatures(dataLoader, model, dict_clss, maxLength=-1):
	realQuantizeds = list()
	realFeatures = list()
	realLabels = list()

	for im, cls_im in tqdm(dataLoader, leave=False, desc='extractFeatures()', ncols=50, bar_format="{desc}|{bar}|{percentage:3.0f}% {elapsed}"):

		im = im.float().cuda()

		# Image embedding into a semantic space
		x, hard, codes = model.encode(im)

		# Accumulate sketch embedding
		realQuantizeds.append(hard.cpu())
		realFeatures.append(x.cpu())

		cls_numeric = torch.from_numpy(utils.numeric_classes(cls_im, dict_clss)).long().cpu()

		realLabels.append(cls_numeric)
	realQuantizeds = torch.cat(realQuantizeds, 0)
	realFeatures = torch.cat(realFeatures, 0)
	realLabels = torch.cat(realLabels, 0)
	if maxLength > 0 and len(realFeatures > maxLength):
		randIdx = torch.randperm(len(realFeatures))[:maxLength]
		realFeatures = realFeatures[randIdx]
		realQuantizeds = realQuantizeds[randIdx]
		realLabels = realLabels[randIdx]
	else:
		randIdx = torch.randperm(len(realFeatures))
		realFeatures = realFeatures[randIdx]
		realQuantizeds = realQuantizeds[randIdx]
		realLabels = realLabels[randIdx]

	return realFeatures, realQuantizeds, realLabels


@torch.no_grad()
def extractOriginalFeatures(dataLoader, model, dict_clss, maxLength=-1):
	realQuantizeds = list()
	realFeatures = list()
	realLabels = list()

	for im, cls_im in tqdm(dataLoader, leave=False, desc='extractFeatures()', ncols=50, bar_format="{desc}|{bar}|{percentage:3.0f}% {elapsed}"):

		im = im.float().cuda()

		# Image embedding into a semantic space
		x, hard, codes = model(im, -1)

		# Accumulate sketch embedding
		realQuantizeds.append(hard.cpu())
		realFeatures.append(x.cpu())

		cls_numeric = torch.from_numpy(utils.numeric_classes(cls_im, dict_clss)).long().cpu()

		realLabels.append(cls_numeric)
	realQuantizeds = torch.cat(realQuantizeds, 0)
	realFeatures = torch.cat(realFeatures, 0)
	realLabels = torch.cat(realLabels, 0)
	if maxLength > 0 and len(realFeatures > maxLength):
		randIdx = torch.randperm(len(realFeatures))[:maxLength]
		realFeatures = realFeatures[randIdx]
		realQuantizeds = realQuantizeds[randIdx]
		realLabels = realLabels[randIdx]
	else:
		randIdx = torch.randperm(len(realFeatures))
		realFeatures = realFeatures[randIdx]
		realQuantizeds = realQuantizeds[randIdx]
		realLabels = realLabels[randIdx]

	return realFeatures, realQuantizeds, realLabels

def selectSubSet(features, quantizeds, labels, nclass):
	# features = torch.from_numpy(features)
	# quantizeds = torch.from_numpy(quantizeds)
	# labels = torch.from_numpy(labels)
	if isinstance(nclass, int):
		allLabels = torch.unique(labels)
		randIdx = torch.randperm(len(allLabels))[:nclass]
		selectedFeatures = list()
		selectedQuantizeds = list()
		selectedLabels = list()
		for idx in randIdx:
			targetLabel = allLabels[idx]
			selectedFeatures.append(features[labels == targetLabel])
			selectedQuantizeds.append(quantizeds[labels == targetLabel])
			selectedLabels.append(labels[labels == targetLabel])
		return torch.cat(selectedFeatures), torch.cat(selectedQuantizeds), torch.cat(selectedLabels)
	else:
		selectedFeatures = list()
		selectedQuantizeds = list()
		selectedLabels = list()
		for targetLabel in nclass:
			selectedFeatures.append(features[labels == targetLabel])
			selectedQuantizeds.append(quantizeds[labels == targetLabel])
			selectedLabels.append(labels[labels == targetLabel])
		return torch.cat(selectedFeatures), torch.cat(selectedQuantizeds), torch.cat(selectedLabels)



def draw_scatters(features, quantizeds, labels, label_to_text_dict, metainfo, filename, sphere=False):
	uniqueLabels = np.unique(labels)
	cmap = np.array(sns.color_palette("tab10", n_colors=len(uniqueLabels)))

	plt.figure(figsize=(8.27,6.5), dpi=384)

	texts = dict()

	ax = plt.gca()
	# remap labels
	for i, l in enumerate(uniqueLabels):
		labels[labels == l] = i
		texts[i] = label_to_text_dict[int(l)]

	if sphere:
		x = np.sin(features[:, 0]) * np.cos(features[:, 1])
		y = np.sin(features[:, 0]) * np.sin(features[:, 1])
		z = np.cos(features[:, 0])
		x = np.arctan2(x, y)
		y = -np.arccos(z)
	else:
		x, y = features[:, 0], features[:, 1]

	ax.scatter(x, y, marker="o", c=cmap[labels], s=18, alpha=0.75, linewidth=0)


	legends = list()
	legned_labels = list()
	# Legend points
	for k, v in texts.items():
		legends.append(Line2D([], [], color="white", marker='o', markerfacecolor=cmap[k]))
		legned_labels.append(v)
	plt.legend(legends, legned_labels, prop={'size': 20}, loc="upper right")

	plt.gca().set_xlabel(metainfo, fontsize=22)

	ax.tick_params(which='both', direction='in', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False, grid_alpha=0.3, labelsize=9, color="#000000")
	ax.grid(True, which="minor", axis="both", lw=0.3, c="#aaaaaa")
	ax.grid(True, which="major", axis="both")
	ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
	ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
	ax.set_axisbelow(True)


	ax.spines['bottom'].set_color('#00000080')
	ax.spines['top'].set_color('#00000080')
	ax.spines['right'].set_color('#00000080')
	ax.spines['left'].set_color('#00000080')



	# plt.tight_layout()

	# plt.gca().set_aspect('equal')
	plt.savefig("{}.pdf".format(filename), transparent=True, bbox_inches="tight")

	plt.clf()
	plt.close()

	return texts

def draw_scatters_by_centers(features, textLabels, hdfeatures, hdlabels, hdlabels_text_map, sphere=False):
	uniqueHDLabels = np.unique(hdlabels)
	remapeed_hdlabels_text_map = dict()
	for i, l in enumerate(uniqueHDLabels):
		hdlabels[hdlabels == l] = i
		remapeed_hdlabels_text_map[i] = hdlabels_text_map[l]

	cmap = palette # np.array(sns.color_palette("Spectral", n_colors=len(uniqueHDLabels)))
	cmapCenters = np.array(sns.color_palette("crest", n_colors=len(textLabels)))

	plt.figure(figsize=(8.27,6.5), dpi=384)

	ax = plt.gca()

	# Plot centers
	if sphere:
		x = np.sin(features[:, 0]) * np.cos(features[:, 1])
		y = np.sin(features[:, 0]) * np.sin(features[:, 1])
		z = np.cos(features[:, 0])
		x = np.arctan2(x, y)
		y = -np.arccos(z)
	else:
		x, y = features[:, 0], features[:, 1]
	# ax.scatter(x, y, marker="*", c="black", s=60, linewidth=0, zorder=999)

	# plot voronoi cell
	vor = Voronoi(features)
	voronoi_plot_2d(vor, ax, show_points=False, show_vertices=False, line_alpha=0.5)



	ax.tick_params(which='both', direction='in', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False, grid_alpha=0.3, labelsize=9, color="#000000")
	ax.grid(False, which="minor", axis="both")
	ax.grid(False, which="major", axis="both")
	ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
	ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
	ax.set_axisbelow(True)


	ax.spines['bottom'].set_color('#00000080')
	ax.spines['top'].set_color('#00000080')
	ax.spines['right'].set_color('#00000080')
	ax.spines['left'].set_color('#00000080')


	texts = list()
	# Annotate centers
	for i in range(len(textLabels)):
		t = ax.text(x[i], y[i], textLabels[i], color="black", ha="left", va="center", size=14, bbox=dict(boxstyle="Round,pad=0.2", fc="white", ec="white", lw=0, alpha=0.9), zorder=999)
		# t.set_bbox(dict(facecolor='white', alpha=0.2, edgecolor='white'))
		texts.append(t)

	adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))




	def draw_hd():
		# Plot unseen category features
		if sphere:
			x = np.sin(hdfeatures[:, 0]) * np.cos(hdfeatures[:, 1])
			y = np.sin(hdfeatures[:, 0]) * np.sin(hdfeatures[:, 1])
			z = np.cos(hdfeatures[:, 0])
			x = np.arctan2(x, y)
			y = -np.arccos(z)
		else:
			x, y = hdfeatures[:, 0], hdfeatures[:, 1]

		ax.scatter(x, y, marker="o", c=cmap[hdlabels], s=20, linewidth=0)

	draw_hd()

	legends = list()
	legned_labels = list()
	# Legend unssen points
	for k, v in remapeed_hdlabels_text_map.items():
		legends.append(Line2D([], [], color="white", marker='o', markerfacecolor=cmap[k]))
		legned_labels.append(v)
	plt.legend(legends, legned_labels, prop={'size': 14}, loc="lower left", bbox_to_anchor=(-0.278, 0))


	# plt.tight_layout()

	# plt.gca().set_aspect('equal')

	plt.savefig("attention_scatter.pdf", transparent=True, bbox_inches="tight")

	plt.clf()
	plt.close()




def draw_scatters_domains(features, quantizeds, labels, hdfeatures, hdquantizeds, hdlabels, class_num, seendomain, hddomain, is_domain=True, sphere=False):
	uniqueLabels = np.unique(labels)
	uniqueHDLabels = np.unique(hdlabels)
	cmap = np.array(sns.color_palette("Spectral", n_colors=max(len(uniqueLabels), len(uniqueHDLabels))))

	plt.figure(figsize=(8.27,6.5), dpi=384)

	ax = plt.gca()

	# remap labels
	for i, l in enumerate(uniqueLabels):
		labels[labels == l] = i


	for i, l in enumerate(uniqueHDLabels):
		hdlabels[hdlabels == l] = i

	if sphere:
		x = np.sin(features[:, 0]) * np.cos(features[:, 1])
		y = np.sin(features[:, 0]) * np.sin(features[:, 1])
		z = np.cos(features[:, 0])
		x = np.arctan2(x, y)
		y = -np.arccos(z)
	else:
		x, y = features[:, 0], features[:, 1]
	ax.scatter(x, y, marker="o" if is_domain else "^", c=cmap[labels], s=12, alpha=0.25, linewidth=0)


	if sphere:

		x = np.sin(hdfeatures[:, 0]) * np.cos(hdfeatures[:, 1])
		y = np.sin(hdfeatures[:, 0]) * np.sin(hdfeatures[:, 1])
		z = np.cos(hdfeatures[:, 0])
		x = np.arctan2(x, y)
		y = -np.arccos(z)
	else:
		x, y = hdfeatures[:, 0], hdfeatures[:, 1]

	ax.scatter(x, y, marker="o" if is_domain else "^", c=cmap[hdlabels], s=12, alpha=1, linewidth=0)




	ax.tick_params(which='both', direction='in', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False, grid_alpha=0.3, labelsize=9, color="#000000")
	ax.grid(True, which="minor", axis="both", lw=0.3, c="#aaaaaa")
	ax.grid(True, which="major", axis="both")
	ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
	ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
	ax.set_axisbelow(True)


	ax.spines['bottom'].set_color('#00000080')
	ax.spines['top'].set_color('#00000080')
	ax.spines['right'].set_color('#00000080')
	ax.spines['left'].set_color('#00000080')


	if is_domain:
		seen = Line2D([], [], color="white", marker='o', markerfacecolor="gray")
		unseen = Line2D([], [], color="white", marker='o', markerfacecolor="black")
		plt.legend([seen, unseen], [f"{seendomain.capitalize()} (seen)", f"{hddomain.capitalize()} (hold-out)"], prop={'size': 11}, loc="upper left")
	else:
		seen = Line2D([], [], color="white", marker='^', ms=10, markerfacecolor="#00000040")
		unseen = Line2D([], [], color="white", marker='^', ms=10, markerfacecolor="black")
		plt.legend([seen, unseen], [f"Seen class", f"Unknown class"], prop={'size': 11}, loc="upper left")


	# plt.tight_layout()
	# plt.gca().set_aspect('equal')

	plt.savefig("{}_{}.pdf".format(seendomain, hddomain), transparent=True, bbox_inches="tight")

	plt.clf()
	plt.close()


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

def dimension_reduction(features, quantizeds, labels, sphere=False, min_dist=0.1):
	features = features.cpu().numpy()
	quantizeds = quantizeds.cpu().numpy()
	labels = labels.cpu().numpy()

	print("UMAP start")
	fit = umap.UMAP(low_memory=False, output_metric="haversine" if sphere else "euclidean", n_neighbors=100, min_dist=min_dist)
	quantizeds = fit.fit_transform(quantizeds)
	features = fit.transform(features)
	print("UMAP end")
	return fit, features, quantizeds, labels



def equal_size_result(features, quantizeds, labels, minCount=500, maxCount=1000):
	randIdx = torch.randperm(len(features)).tolist()
	uniqueLabels = torch.unique(labels).tolist()
	allCounts = list()
	for l in uniqueLabels:
		count = (labels == l).sum()
		allCounts.append(int(count))
	targetCount = min(max(min(allCounts), minCount), maxCount)
	currCount = {
		l: 0 for l in uniqueLabels
	}
	filteredFeatures = list()
	filteredQuantizeds = list()
	filteredLabels = list()
	for i in tqdm(randIdx, leave=False, desc='equal_size_result()', ncols=50, bar_format="{desc}|{bar}|{percentage:3.0f}% {elapsed}"):
		if currCount[int(labels[i])] > targetCount:
			continue
		filteredFeatures.append(features[i])
		filteredQuantizeds.append(quantizeds[i])
		filteredLabels.append(labels[i])
		currCount[int(labels[i])] += 1
	return torch.stack(filteredFeatures), torch.stack(filteredQuantizeds), torch.stack(filteredLabels)
