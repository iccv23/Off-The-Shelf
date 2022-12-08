import numpy as np
import torch
import warnings


from tqdm import tqdm




def _mAP_NP(database_hash, test_hash, database_labels, test_labels, args):  # R = 1000
    # binary the hash code
    R = args.R
    T = args.T
    database_hash = database_hash.astype(np.int32) * 2 - 1
    test_hash = test_hash.astype(np.int32) * 2 - 1

    query_num = test_hash.shape[0]  # total number for testing
    sim = np.dot(database_hash, test_hash.T)
    ids = np.argsort(-sim, axis=0)
    #data_dir = 'data/' + args.data_name
    #ids_10 = ids[:10, :]

    #np.save(data_dir + '/ids.npy', ids_10)
    APx = []
    Recall = []

    for i in tqdm(range(query_num), desc="mAP", leave=False, ncols=50, bar_format="{desc}|{bar}|{percentage:3.0f}% {elapsed}"):  # for i=0
        label = test_labels[i, :]  # the first test labels
        if np.sum(label) == 0:  # ignore images with meaningless label in nus wide
            continue
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)  #

        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
        if relevant_num == 0:  # even no relevant image, still need add in APx for calculating the mean
            APx.append(0)

        all_relevant = np.sum(database_labels == label, axis=1) > 0
        all_num = np.sum(all_relevant)
        r = relevant_num / all_num.astype(float)
        Recall.append(r)

    return np.mean(np.array(APx)), np.mean(np.array(Recall)), APx


def mean_average_precision(database_hash: torch.FloatTensor, test_hash: torch.FloatTensor, database_labels: torch.BoolTensor, test_labels: torch.BoolTensor, K):
    AP = list()
    Recall = list()
    Precision = list()
    allIds = list()
    warnings.warn("mAP by torch is 1%% lower than numpy version.")
    for queryX, queryLabels in tqdm(zip(test_hash, test_labels), leave=False, desc='MAP', ncols=50, bar_format="{desc}|{bar}|{percentage:3.0f}% {elapsed}"):
        thisAP, thisR, thisPrecision, ids = _partedMAP(database_hash, queryX[None, :], database_labels, queryLabels[None, :], K)
        AP.append(thisAP)
        Recall.append(thisR)
        Precision.append(thisPrecision)
        allIds.append(ids)
    # # appen remaining AP
    # thisAP, thisR = _partedMAP(database_hash[batches * 64:], test_hash, database_labels[batches * 64:], test_labels, args)
    # AP.append(thisAP)
    # R.append(thisR)

    AP = torch.cat(AP)
    Recall = torch.cat(Recall)
    Precision = torch.cat(Precision)
    allIds = torch.stack(allIds)
    return AP, Recall, Precision, allIds

def mean_average_precision_bits(database_hash: torch.FloatTensor, test_hash: torch.FloatTensor, database_labels: torch.BoolTensor, test_labels: torch.BoolTensor, K):
    AP = list()
    Recall = list()
    Precision = list()
    allIds = list()
    warnings.warn("mAP by torch is 1%% lower than numpy version.")
    for queryX, queryLabels in tqdm(zip(test_hash, test_labels), leave=False, desc='MAP', ncols=50, bar_format="{desc}|{bar}|{percentage:3.0f}% {elapsed}"):
        thisAP, thisR, thisPrecision, ids = _partedMAP_bits(database_hash, queryX[None, :], database_labels, queryLabels[None, :], K)
        AP.append(thisAP)
        Recall.append(thisR)
        Precision.append(thisPrecision)
        allIds.append(ids)
    # # appen remaining AP
    # thisAP, thisR = _partedMAP(database_hash[batches * 64:], test_hash, database_labels[batches * 64:], test_labels, args)
    # AP.append(thisAP)
    # R.append(thisR)

    AP = torch.cat(AP)
    Recall = torch.cat(Recall)
    Precision = torch.cat(Precision)
    allIds = torch.stack(allIds)
    return AP, Recall, Precision, allIds


def _partedMAP_bits(database_hash: torch.FloatTensor, test_hash: torch.FloatTensor, database_labels: torch.BoolTensor, test_labels: torch.BoolTensor, K):  # R = 1000
    R = K

    # [1, Nb]
    sim = test_hash @ database_hash.T

    # [Nq, R] top R queried from base
    _, ids = torch.topk(sim, R, -1, largest=True, sorted=True)

    # [Nq, R, nclass]
    queried_labels = database_labels[ids]
    # [Nq, R, nclass] [Nq, 1, nclass] -> [Nq, R] ordered matching result
    matched = (torch.logical_and(queried_labels, test_labels[:, None]).sum(-1) > 0).float()
    # [Nq] Does this query has any match?
    hasMatched = matched.sum(-1)
    # cum-sum along R dim
    L = torch.cumsum(matched, -1)
    # [Nq, R]
    P = L / torch.arange(1, R + 1, 1, device=L.device, dtype=torch.float)

    # [Nq] / [Nq]
    AP = (P * matched).sum(-1) / hasMatched
    # for results has no match, set to 0
    AP[hasMatched < 1] = 0

    # [Nq, Nb] -> [Nq], Recall base
    allRelevent = (torch.logical_and(test_labels[:, None], database_labels).sum(-1) > 0).sum(-1).float()

    # [Nq]
    Recall = hasMatched / allRelevent

    # [Nq]
    Precision = P[:, -1]

    # [Nq]
    return AP, Recall, Precision, ids


def _partedMAP(database_hash: torch.FloatTensor, test_hash: torch.FloatTensor, database_labels: torch.BoolTensor, test_labels: torch.BoolTensor, K):  # R = 1000
    R = K

    # [1, Nb]
    sim = -((test_hash - database_hash) ** 2).sum(-1)

    # [Nq, R] top R queried from base
    _, ids = torch.topk(sim, R, -1, largest=True, sorted=True)

    # [Nq, R, nclass]
    queried_labels = database_labels[ids]
    # [Nq, R, nclass] [Nq, 1, nclass] -> [Nq, R] ordered matching result
    matched = (torch.logical_and(queried_labels, test_labels[:, None]).sum(-1) > 0).float()
    # [Nq] Does this query has any match?
    hasMatched = matched.sum(-1)
    # cum-sum along R dim
    L = torch.cumsum(matched, -1)
    # [Nq, R]
    P = L / torch.arange(1, R + 1, 1, device=L.device, dtype=torch.float)

    # [Nq] / [Nq]
    AP = (P * matched).sum(-1) / hasMatched
    # for results has no match, set to 0
    AP[hasMatched < 1] = 0

    # [Nq, Nb] -> [Nq], Recall base
    allRelevent = (torch.logical_and(test_labels[:, None], database_labels).sum(-1) > 0).sum(-1).float()

    # [Nq]
    Recall = hasMatched / allRelevent

    # [Nq]
    Precision = P[:, -1]

    # [Nq]
    return AP, Recall, Precision, ids


@torch.no_grad()
def get_rank_list(database_hash: torch.BoolTensor, test_hash: torch.BoolTensor, database_labels: torch.BoolTensor, test_labels: torch.BoolTensor):
    precisions = list()
    recalls = list()
    pAtH2s = list()
    database_hash = database_hash.float() * 2 - 1
    for queryX, queryLabels in tqdm(zip(test_hash, test_labels), leave=False, desc='MAP', ncols=50, bar_format="{desc}|{bar}|{percentage:3.0f}% {elapsed}"):
        # [1, Nb]
        precision, recall, pAtH2 = _partedRank(database_hash, queryX[None, :], database_labels, queryLabels[None, :])
        precisions.append(precision)
        recalls.append(recall)
        pAtH2s.append(pAtH2)
    # [Nb], [Nb], float
    return torch.cat(precisions).mean(0), torch.cat(recalls).mean(0), float(torch.tensor(pAtH2s).mean())


def _partedRank(database_hash: torch.FloatTensor, test_hash: torch.BoolTensor, database_labels: torch.BoolTensor, test_labels: torch.BoolTensor):  # R = all
    # [1, Nb]
    sim = (test_hash.float() * 2 - 1) @ database_hash.T

    bits = test_hash.shape[-1]
    h2 = bits - 4

    # [1, Nb] queried from base
    values, ids = torch.sort(sim, -1, descending=True)

    # the first index that distance > 2
    rankinsideH2 = torch.nonzero((values < h2)[0])[0]

    # [1, R, nclass]
    queried_labels = database_labels[ids]
    # [1, R, nclass] [1, 1, nclass] -> [1, R] ordered matching result
    matched = (torch.logical_and(queried_labels, test_labels[:, None]).sum(-1) > 0).float()
    cumsum = matched.cumsum(-1)
    # [1, R]
    precision = (cumsum / torch.arange(1, matched.shape[-1] + 1, 1, device=matched.device, dtype=torch.float))
    # [1, R]
    recall = (cumsum / matched.sum(-1, keepdim=True))



    # [1, R], [1, R], float
    return precision, recall, float(precision[0, rankinsideH2])
