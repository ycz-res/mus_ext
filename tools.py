import h5py
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

# 单长度集或混合集（有 seq_len）截断上限；不需要截断可设 None。
MAX_SEQ_LEN = 1000
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle


def load_data(file_path, model_select_, tmp_flag, loss_flag):
    """
    :param file_path:
    :param model_select_: 选择模型类型
    :param tmp_flag: label的最小值，用于将label转化为0开始;如果为-1，则取最小值作为 0
    :return: X, Y, n_class, x_len, x_dim, label_flag, is_varlen
             is_varlen 为 True 时 X 为 list，每项 (C, L_i)；需用 VarLenSeqDataset + collate_varlen_cnn
    """
    mat_file = h5py.File(file_path, 'r')  # 加载.mat数据文件
    X_load, Y_load = mat_file['data'], mat_file['label_num']
    seq_len = None
    if 'seq_len' in mat_file:
        seq_len = np.array(mat_file['seq_len']).squeeze().astype(np.int64)
    X_load = np.array(X_load).transpose(2, 1, 0)[:, :, 1].squeeze()
    Y_load = np.array(Y_load).squeeze()
    if Y_load.ndim == 2:
        if Y_load.shape[0] < Y_load.shape[1]:
            Y_load = Y_load[0]
        else:
            Y_load = Y_load[:, 0]
    index = Y_load >= 5  # todo 截断 从5个unit开始预测
    X_load, Y_load = X_load[index,], Y_load[index]
    if seq_len is not None:
        seq_len = seq_len[index]

    # 预处理 (排序->归一化->特征提取->标签转换)
    [X, y1, label_flag] = preprocess(X_load, Y_load, np.array([0, 1, 2]), tmp_flag, seq_len=seq_len)  # todo 选择阶数

    if loss_flag == 'CrossEntropy':
        n_class_ = len(np.unique(y1))  # 模型输出维度
    else:
        n_class_ = 1

    if isinstance(X, list):
        is_varlen = True
        x_dim_ = int(X[0].shape[-1])
        x_len_ = max(int(x.shape[0]) for x in X)
        if 'CNN' in model_select_ or model_select_ in (
            'ResCNN', 'CNNsimple', 'BiLSTM_GAMP', 'Transformer_CLS',
        ):
            X = [np.transpose(x, (1, 0)).astype(np.float32) for x in X]
        elif model_select_ == 'FCN':
            raise NotImplementedError("变长序列暂不支持 FCN，请使用 ResCNN 或固定长度 .mat")
        else:
            raise NotImplementedError(f"变长序列与 model_select={model_select_} 未适配")
    else:
        is_varlen = False
        x_len_ = X.shape[-2]
        x_dim_ = X.shape[-1]
        if 'CNN' in model_select_ or model_select_ in (
            'ResCNN', 'CNNsimple', 'BiLSTM_GAMP', 'Transformer_CLS',
        ):
            X = X.swapaxes(1, 2)
        elif model_select_ == 'FCN':
            X = X.reshape(-1, x_dim_)

    Y = y1.astype(np.float32)

    return X, Y, n_class_, x_len_, x_dim_, label_flag, is_varlen

def reverse_label(y, a):
    out = y + a
    return out

def load_real_data(file_path, model_select_, tmp_flag, loss_flag):
    """
    :param file_path:
    :param model_select_: 选择模型类型
    :param tmp_flag: label的最小值，用于将label转化为0开始;如果为-1，则取最小值作为 0
    :return: 与 load_data 相同，最后一项为 is_varlen
    """
    mat_file = h5py.File(file_path, 'r')  # 加载.mat数据文件
    X_load, Y_load = mat_file['data'], mat_file['label_num']
    seq_len = None
    if 'seq_len' in mat_file:
        seq_len = np.array(mat_file['seq_len']).squeeze().astype(np.int64)
    X_load = np.array(X_load).transpose(2, 1, 0)[:, :, 1].squeeze()
    Y_load = np.array(Y_load).squeeze()
    if Y_load.ndim == 2:
        if Y_load.shape[0] < Y_load.shape[1]:
            Y_load = Y_load[0]
        else:
            Y_load = Y_load[:, 0]
    index = Y_load >= 5  # todo 截断 从20个unit开始预测
    X_load, Y_load = X_load[index,], Y_load[index]
    if seq_len is not None:
        seq_len = seq_len[index]

    # 预处理 (排序->归一化->特征提取->标签转换)
    [X, y1, label_flag] = preprocess(X_load, Y_load, np.array([0, 1, 2]), tmp_flag, seq_len=seq_len)  # todo 选择阶数

    # X_load, Y_load = X_load[:1000, ], Y_load[:1000]  # todo 截断
    # noise_load, amp_load = noise_load[:1000], amp_load[:1000]
    # thr_load, thr_var_load = thr_load[:1000], thr_var_load[:1000]

    if loss_flag == 'CrossEntropy':
        n_class_ = len(np.unique(y1))  # 模型输出维度
    else:
        n_class_ = 1

    if isinstance(X, list):
        is_varlen = True
        x_dim_ = int(X[0].shape[-1])
        x_len_ = max(int(x.shape[0]) for x in X)
        if 'CNN' in model_select_ or model_select_ in (
            'ResCNN', 'CNNsimple', 'BiLSTM_GAMP', 'Transformer_CLS',
        ):
            X = [np.transpose(x, (1, 0)).astype(np.float32) for x in X]
        elif model_select_ == 'FCN':
            raise NotImplementedError("变长序列暂不支持 FCN，请使用 ResCNN 或固定长度 .mat")
        else:
            raise NotImplementedError(f"变长序列与 model_select={model_select_} 未适配")
    else:
        is_varlen = False
        x_len_ = X.shape[-2]
        x_dim_ = X.shape[-1]
        if 'CNN' in model_select_ or model_select_ in (
            'ResCNN', 'CNNsimple', 'BiLSTM_GAMP', 'Transformer_CLS',
        ):
            X = X.swapaxes(1, 2)
        elif model_select_ == 'FCN':
            X = X.reshape(-1, x_dim_)

    Y = y1.astype(np.float32)

    return X, Y, n_class_, x_len_, x_dim_, label_flag, is_varlen


def set_seed(seed_value):
    np.random.RandomState(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def trans_label(y, a):
    out = (y - a).astype(np.int64)
    # assert out.min() == 0
    return out

def n_order_diff(x, b):
    for _ in range(b):
        x = np.diff(x, axis=-1)  # 使用NumPy的diff函数计算一阶差分
    return x


def _preprocess_one_sample_1d(x_1d: np.ndarray, n_order_array: np.ndarray) -> np.ndarray:
    """单条波形 (L,) -> 特征 (L, C)，与 x_preprocess 整批逻辑一致。"""
    x = x_1d.reshape(1, -1)
    X_out = x_preprocess(x, 3)
    X_out = X_out[:, :, n_order_array]
    return X_out[0].astype(np.float32)


def x_preprocess(x, b):
    if b == 0:
        out = x[:, :, np.newaxis]
    else:
        out = [x]
        for b_i in range(1, b + 1):
            dx = n_order_diff(x, b_i)
            dx = np.concatenate((np.zeros((dx.shape[0], b_i)), dx), axis=-1)
            out.append(dx)
        out = np.stack(out, axis=-1)
    return out.astype(np.float32)


def preprocess(X_, Y_, n_order_array, label_flag, seq_len=None):
    """
    :param X_: (N, L)
    :param Y_: (N,)
    :param n_order_array: e.g. np.array([0,1,2]) : 0 - 原始数据，1 - 一阶差分，2 - 二阶差分
    :param label_flag: 标签的最小值，用于将label转化为0开始;如果为-1，则取最小运动单位数量作为 0
    :param seq_len: 可选，(N,) 每条样本有效长度（混合 .mat 由 mix_datasets 写入）
    :return: X_out 为 (N, L, C) ndarray，或 seq_len 存在时为 list[np.ndarray]，每项形状 (L_i, C)
    """
    assert len(X_.shape) == 2 and len(X_) == len(Y_)

    if seq_len is None:
        # 从小到大排序
        if X_[:, :10].sum() > X_[:, -10:].sum():
            X_out = np.flip(X_, axis=-1)  # (N,L)
        else:
            X_out = X_

        # 归一化
        L_dim = X_out.shape[-1]
        fen = np.repeat(X_out.max(axis=-1, keepdims=True), L_dim, axis=-1)
        X_out = np.divide(X_out, fen)

        if MAX_SEQ_LEN is not None and X_out.shape[-1] > MAX_SEQ_LEN:
            X_out = X_out[:, :MAX_SEQ_LEN]

        # 特征提取（长度随输入序列长度动态变化）
        X_out = x_preprocess(X_out, 3)
        X_out = X_out[:, :, n_order_array]  # todo 输入特征维度
    else:
        seq_len_np = np.asarray(seq_len).reshape(-1).astype(np.int64)
        assert len(seq_len_np) == len(X_)
        if MAX_SEQ_LEN is not None:
            seq_len_np = np.minimum(seq_len_np, MAX_SEQ_LEN)

        out_list: list[np.ndarray] = []
        for i in range(len(X_)):
            Li = int(seq_len_np[i])
            if Li <= 0:
                raise ValueError(f"seq_len[{i}]={Li} 无效")
            xi = np.asarray(X_[i, :Li], dtype=np.float32)
            if Li >= 10:
                if xi[:10].sum() > xi[-10:].sum():
                    xi = np.flip(xi)
            elif Li >= 2 and xi[0] > xi[-1]:
                xi = np.flip(xi)
            m = float(np.max(xi))
            if m <= 0:
                m = 1.0
            xi = xi / m
            out_list.append(_preprocess_one_sample_1d(xi, n_order_array))

        X_out = out_list

    # 标签转换
    if label_flag < 0:
        label_flag = Y_.min()
    Y_out = trans_label(Y_, label_flag)

    return X_out, Y_out, label_flag


class VarLenSeqDataset(Dataset):
    """每条样本 (C, L_i)，长度可不同；与 collate_varlen_cnn 配合使用。"""

    def __init__(self, xs: list, y: np.ndarray):
        self.xs = xs
        self.y = np.asarray(y)

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int):
        x = self.xs[idx]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y


def collate_varlen_cnn(batch):
    """将 list[(C, L_i), y] pad 为 (B, C, L_max)，仅 pad 本 batch 最大长度。"""
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[1] for x in xs], dtype=torch.long)
    c = int(xs[0].shape[0])
    max_l = int(lengths.max().item())
    b = len(xs)
    padded = torch.zeros(b, c, max_l, dtype=torch.float32)
    for i, x in enumerate(xs):
        li = int(x.shape[1])
        padded[i, :, :li] = x
    y = torch.stack(ys, dim=0)
    return padded, y


def collate_varlen_cnn_fake_aux(batch):
    """与 main.test(fake) 的 7 元组 DataLoader 一致，后 5 列为占位 0。"""
    padded, y = collate_varlen_cnn(batch)
    b = padded.shape[0]
    z = torch.zeros(b, dtype=torch.float32)
    return padded, y, z, z, z, z, z


def fit_label_standardize(y_train: np.ndarray) -> tuple[float, float]:
    """在训练集标签上拟合 (mean, std)，用于 MSE 回归。"""
    y = np.asarray(y_train, dtype=np.float64).ravel()
    mean = float(y.mean())
    std = float(y.std())
    if std < 1e-8:
        std = 1.0
    return mean, std


def apply_label_standardize(y: np.ndarray, mean: float, std: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    return ((y - mean) / std).astype(np.float32)


def inverse_label_standardize(y: np.ndarray, mean: float, std: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    return (y * std + mean).astype(np.float32)


def save_label_norm_npz(path: Path, mean: float, std: float, enabled: bool) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=np.float64(mean), std=np.float64(std), enabled=np.bool_(enabled))


def load_label_norm_npz(path: Path) -> tuple[float | None, float | None, bool]:
    """返回 (mean, std, enabled)；未启用时 mean/std 为 None。"""
    path = Path(path)
    if not path.exists():
        return None, None, False
    z = np.load(path)
    en = bool(np.asarray(z["enabled"]).item())
    if not en:
        return None, None, False
    return float(z["mean"]), float(z["std"]), True
