import numpy as np
from torch.utils.data import Dataset
import torch
import math


class ClassiregressionDataset(Dataset):

    def __init__(self, data, indices):
        super(ClassiregressionDataset, self).__init__()

        self.data = data  # this is a subclass of the BaseData class in data.py
        self.IDs = indices  # list of data IDs, but also mapping between integer index and ID
        self.feature_df = self.data.feature_df.loc[self.IDs]
        self.feature_df_label = self.data.feature_df_label.loc[self.IDs]

        #self.labels_dfm = self.data.labels_dfm.loc[self.IDs]
        #self.labels_dfa = self.data.labels_dfa.loc[self.IDs]


    def __getitem__(self, ind):
        """
        For a given integer index, returns the corresponding (seq_length, feat_dim) array and a noise mask of same shape
        Args:
            ind: integer index of sample in dataset
        Returns:
            X: (seq_length, feat_dim) tensor of the multivariate time series corresponding to a sample
            y: (seq_length, 2) tensor of labels (num_labels > 1 for multi-task models) for each sample
            ID: ID of sample
        """

        X = self.feature_df.loc[self.IDs[ind]].values  # (seq_length, feat_dim) array
        #ym = self.labels_dfm.loc[self.IDs[ind]].values  # (num_labels,) array
        #ya = self.labels_dfa.loc[self.IDs[ind]].values  # (num_labels,) array
        Y = self.feature_df_label.loc[self.IDs[ind]].values  # (seq_length, feat_dim) array

        #return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(ya), self.IDs[ind]

        return torch.from_numpy(X), torch.from_numpy(Y), self.IDs[ind]
        dim=5
        track_diff = [[] for _ in range(0, dim)]
        for d in range(0, dim):
            # for j in range(len(X)):
            traj=X[:, :2]
            for t in range(self.data.track_id_len[self.IDs[ind]]):
                tmpx = traj[t + int(d / 2) + 1] - traj[t]
                track_diff[d].append(tmpx)
        
        X1 = np.asarray(track_diff)
        return torch.from_numpy(X1), torch.from_numpy(Y), self.IDs[ind]

    def __len__(self):
        return len(self.IDs)


class ClassiregressionDataset_no_data_difference(Dataset):

    def __init__(self, data, indices):
        super(ClassiregressionDataset_no_data_difference, self).__init__()

        self.data = data  # this is a subclass of the BaseData class in data.py
        self.IDs = indices  # list of data IDs, but also mapping between integer index and ID
        self.feature_df = self.data.feature_df_original.loc[self.IDs]
        self.feature_df_label = self.data.feature_df_label.loc[self.IDs]

        #self.labels_dfm = self.data.labels_dfm.loc[self.IDs]
        #self.labels_dfa = self.data.labels_dfa.loc[self.IDs]


    def __getitem__(self, ind):
        """
        For a given integer index, returns the corresponding (seq_length, feat_dim) array and a noise mask of same shape
        Args:
            ind: integer index of sample in dataset
        Returns:
            X: (seq_length, feat_dim) tensor of the multivariate time series corresponding to a sample
            y: (seq_length, 2) tensor of labels (num_labels > 1 for multi-task models) for each sample
            ID: ID of sample
        """

        X = self.feature_df.loc[self.IDs[ind]].values  # (seq_length, feat_dim) array
        #ym = self.labels_dfm.loc[self.IDs[ind]].values  # (num_labels,) array
        #ya = self.labels_dfa.loc[self.IDs[ind]].values  # (num_labels,) array
        Y = self.feature_df_label.loc[self.IDs[ind]].values  # (seq_length, feat_dim) array

        #return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(ya), self.IDs[ind]

        return torch.from_numpy(X), torch.from_numpy(Y), self.IDs[ind]
        dim=5
        track_diff = [[] for _ in range(0, dim)]
        for d in range(0, dim):
            # for j in range(len(X)):
            traj=X[:, :2]
            for t in range(self.data.track_id_len[self.IDs[ind]]):
                tmpx = traj[t + int(d / 2) + 1] - traj[t]
                track_diff[d].append(tmpx)
        
        X1 = np.asarray(track_diff)
        return torch.from_numpy(X1), torch.from_numpy(Y), self.IDs[ind]

    def __len__(self):
        return len(self.IDs)

def transduct_mask(X, mask_feats, start_hint=0.0, end_hint=0.0):
    """
    Creates a boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """

    mask = np.ones(X.shape, dtype=bool)
    start_ind = int(start_hint * X.shape[0])
    end_ind = max(start_ind, int((1 - end_hint) * X.shape[0]))
    mask[start_ind:end_ind, mask_feats] = 0

    return mask


def compensate_masking(X, mask):
    """
    Compensate feature vectors after masking values, in a way that the matrix product W @ X would not be affected on average.
    If p is the proportion of unmasked (active) elements, X' = X / p = X * feat_dim/num_active
    Args:
        X: (batch_size, seq_length, feat_dim) torch tensor
        mask: (batch_size, seq_length, feat_dim) torch tensor: 0s means mask and predict, 1s: unaffected (active) input
    Returns:
        (batch_size, seq_length, feat_dim) compensated features
    """

    # number of unmasked elements of feature vector for each time step
    num_active = torch.sum(mask, dim=-1).unsqueeze(-1)  # (batch_size, seq_length, 1)
    # to avoid division by 0, set the minimum to 1
    num_active = torch.max(num_active, torch.ones(num_active.shape, dtype=torch.int16))  # (batch_size, seq_length, 1)
    return X.shape[-1] * X / num_active



def collate_superv_for_weight(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    features, labels, IDs = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    lengths = [Y.shape[0] for Y in labels]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    Y = torch.zeros(batch_size, max_len, labels[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        Y[i, :end, :] = labels[i][:end, :]
# JY state
        if Y.shape[2] == 3:
            Y[i, end:, 2] = 5 # set padding position to 5 to calculate cross entropy with ignore_index=5

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep


    #targets_a=Y[:, :, 0]
    #targets_K = Y[:, :, 1]
    targets_a = Y[:, :, 0].numpy()
    targets_k = Y[:, :, 1].numpy()

    #接下来要找每一条轨迹的CP
    #CP_a=[[]for _ in range(len(targets_a))]
    #CP_k = [[] for _ in range(len(targets_k))]

    CP=[]  #存放断点   最后一个的索引

    w_mask= torch.zeros_like(Y)

    #mask[799][199][1]
    #mask[i][j][1]


    for i in range(len(targets_a)):
        unique_elements_a = np.unique(targets_a[i])
        unique_elements_k = np.unique(targets_k[i])

        cp1=[-1]
        seg=[]
        alpha_seg=[0]
        k_seg=[0]

        for alpha in unique_elements_a:
            if alpha!=0:
                indice_tmp=find_last_indices_of_continuous(targets_a[i],alpha)
                cp1=cp1+indice_tmp

        for K_ in unique_elements_k:
            if K_!=0:
                indice_tmp=find_last_indices_of_continuous(targets_k[i],K_)
                cp1=cp1+indice_tmp

        cp1=np.array(cp1)
        cp = np.unique(cp1)

        for ind in range(len(cp)-1):
            seg.append(cp[ind+1]-cp[ind])
            alpha_seg.append(targets_a[i][cp[ind+1]])
            k_seg.append(targets_k[i][cp[ind+1]])

        alpha_seg.append(0)
        k_seg.append(0)

        CP.append(cp)  # 所有轨迹对应的cp

        for t in range(1,len(cp),1):
            for j in range(cp[t-1]+1,cp[t]+1,1):
                len_tmp=seg[t-1]
                weight1=1.5 - (1 / (1 + math.exp(-0.1*abs(len_tmp-20))))


                for_weight2_a1=alpha_seg[t-1]
                for_weight2_a2=alpha_seg[t+1]

                w_for_weight2_a=(abs(alpha_seg[t]-for_weight2_a1)+abs(alpha_seg[t]-for_weight2_a2))/2

                weight2_a= 1.5 - (1 / (1 + math.exp(-1*w_for_weight2_a*min(abs(j-float(cp[t]+0.5)),abs(j-float(cp[t-1]+0.5))))))

                for_weight2_k1 = k_seg[t - 1]
                for_weight2_k2 = k_seg[t + 1]

                w_for_weight2_k = (abs(k_seg[t] - for_weight2_k1) + abs(k_seg[t] - for_weight2_k2)) / 2

                weight2_k = 1.5 - (1 / (1 + math.exp(-1 * w_for_weight2_k * min(abs(j - float(cp[t] + 0.5)), abs(j - float(cp[t - 1] + 0.5))))))

                w_mask[i][j][0]=weight1+weight2_a
                w_mask[i][j][1] = weight1 + weight2_k



    return X, Y, padding_masks, IDs, w_mask


def collate_superv_for_cp(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    features, labels, IDs = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    lengths = [Y.shape[0] for Y in labels]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    Y = torch.zeros(batch_size, max_len, labels[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    CP = torch.zeros((Y.shape[0], Y.shape[1], 1), device=Y.device)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        Y[i, :end, :] = labels[i][:end, :]
        labs = Y[i, :, :].cpu().numpy()
        cp_idx = np.argwhere((labs[:-1, :] != labs[1:, :]).sum(1) != 0).flatten()+1
        CP[i][cp_idx] = 1
        CP[i][end:] = 5
# JY state 
        if Y.shape[2] == 3:
            Y[i, end:, 2] = 5 # set padding position to 5 to calculate cross entropy with ignore_index=5
    Y = torch.cat((Y, CP), dim=-1)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep


    return X, Y, padding_masks, IDs


def collate_superv(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    features, labels, IDs = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    lengths = [Y.shape[0] for Y in labels]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    Y = torch.zeros(batch_size, max_len, labels[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        Y[i, :end, :] = labels[i][:end, :]
# JY state
        if Y.shape[2] == 3:
            Y[i, end:, 2] = 5 # set padding position to 5 to calculate cross entropy with ignore_index=5


    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep

    return X, Y, padding_masks, IDs



def find_last_indices_of_continuous(arr,num):
    last_indices = []
    continuous = False  # 用于跟踪当前是否在连续的2的片段中
    for i in range(len(arr)):
        if arr[i] == num:
            if i == len(arr)-1:
                last_indices.append(i)  # 添加连续片段最后一个2的索引
            if not continuous:  # 如果之前不是连续的2的片段
                continuous = True
                start_index = i  # 记录连续片段的起始索引
            else:
                if i+1 < len(arr) and arr[i+1] != num:  # 如果下一个元素不是2
                    continuous = False  # 标记连续片段结束
                    last_indices.append(i)  # 添加连续片段最后一个2的索引
        else:
            if continuous:  # 如果之前是连续的2的片段
                continuous = False  # 标记连续片段结束
                last_indices.append(i-1)  # 添加连续片段最后一个2的索引

    return last_indices



def collate_unsuperv(data, max_len=None, mask_compensation=False):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, mask).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - mask: boolean torch tensor of shape (seq_length, feat_dim); variable seq_length.
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    """

    batch_size = len(data)
    features, masks, IDs = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    target_masks = torch.zeros_like(X,
                                    dtype=torch.bool)  # (batch_size, padded_length, feat_dim) masks related to objective
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]
        target_masks[i, :end, :] = masks[i][:end, :]

    targets = X.clone()
    X = X * target_masks  # mask input
    if mask_compensation:
        X = compensate_masking(X, target_masks)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep
    target_masks = ~target_masks  # inverse logic: 0 now means ignore, 1 means predict
    return X, targets, target_masks, padding_masks, IDs


def noise_mask(X, masking_ratio, lm=3, mode='separate', distribution='geometric', exclude_feats=None):
    """

    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':  # each variable (feature) is independent
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])

    return mask


def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked
    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
