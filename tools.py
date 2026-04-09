import h5py
import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle


def load_data(file_path, model_select_, tmp_flag, loss_flag):
    """
    :param file_path:
    :param model_select_: 选择模型类型
    :param tmp_flag: label的最小值，用于将label转化为0开始;如果为-1，则取最小值作为 0
    :return:
    """
    mat_file = h5py.File(file_path, 'r')  # 加载.mat数据文件
    X_load, Y_load = mat_file['data'], mat_file['label_num']
    X_load, Y_load = np.array(X_load).transpose(2, 1, 0)[:, :, 1].squeeze(), np.array(Y_load).squeeze()
    index = Y_load.squeeze() >= 5  # todo 截断 从5个unit开始预测
    X_load, Y_load = X_load[index,], Y_load[index]

    # 预处理 (排序->归一化->特征提取->标签转换)
    [X, y1, label_flag] = preprocess(X_load, Y_load, np.array([0, 1, 2]), tmp_flag)  # todo 选择阶数

    if loss_flag == 'CrossEntropy':
        n_class_ = len(np.unique(y1))  # 模型输出维度
    else:
        n_class_ = 1
    x_len_ = X.shape[-2]
    x_dim_ = X.shape[-1]

    Y = y1.astype(np.float32)

    # 正则化
    # X = stander(X)  # todo 正则化

    if 'CNN' in model_select_:
        X = X.swapaxes(1, 2)
    elif model_select_ == 'FCN':
        X = X.reshape(-1, x_dim_)

    return X, Y, n_class_, x_len_, x_dim_, label_flag

def reverse_label(y, a):
    out = y + a
    return out

def load_real_data(file_path, model_select_, tmp_flag, loss_flag):
    """
    :param file_path:
    :param model_select_: 选择模型类型
    :param tmp_flag: label的最小值，用于将label转化为0开始;如果为-1，则取最小值作为 0
    :return:
    """
    mat_file = h5py.File(file_path, 'r')  # 加载.mat数据文件
    X_load, Y_load = mat_file['data'], mat_file['label_num']
    X_load, Y_load = np.array(X_load).transpose(2, 1, 0)[:, :, 1].squeeze(), np.array(Y_load).squeeze()
    index = Y_load.squeeze() >= 5  # todo 截断 从20个unit开始预测
    X_load, Y_load = X_load[index,], Y_load[index]

    # 预处理 (排序->归一化->特征提取->标签转换)
    [X, y1, label_flag] = preprocess(X_load, Y_load, np.array([0, 1, 2]), tmp_flag)  # todo 选择阶数

    # X_load, Y_load = X_load[:1000, ], Y_load[:1000]  # todo 截断
    # noise_load, amp_load = noise_load[:1000], amp_load[:1000]
    # thr_load, thr_var_load = thr_load[:1000], thr_var_load[:1000]

    if loss_flag == 'CrossEntropy':
        n_class_ = len(np.unique(y1))  # 模型输出维度
    else:
        n_class_ = 1
    x_len_ = X.shape[-2]
    x_dim_ = X.shape[-1]

    Y = y1.astype(np.float32)

    # 正则化
    # X = stander(X)  # todo 正则化

    if 'CNN' in model_select_:
        X = X.swapaxes(1, 2)
    elif model_select_ == 'FCN':
        X = X.reshape(-1, x_dim_)

    return X, Y, n_class_, x_len_, x_dim_, label_flag


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


def preprocess(X_, Y_, n_order_array, label_flag):
    """
    :param X_: (N, L)
    :param Y_: (N,)
    :param n_order_array: e.g. np.array([0,1,2]) : 0 - 原始数据，1 - 一阶差分，2 - 二阶差分
    :param label_flag: 标签的最小值，用于将label转化为0开始;如果为-1，则取最小运动单位数量作为 0
    :return:
    """
    assert len(X_.shape) == 2 and len(X_) == len(Y_)

    # 从小到大排序
    if X_[:, :10].sum() > X_[:, -10:].sum():
        X_out = np.flip(X_, axis=-1)  # (N,L)
    else:
        X_out = X_

    # 归一化
    seq_len = X_out.shape[-1]
    fen = np.repeat(X_out.max(axis=-1, keepdims=True), seq_len, axis=-1)
    X_out = np.divide(X_out, fen)

    # 特征提取（长度随输入序列长度动态变化）
    X_out = x_preprocess(X_out, 3)
    X_out = X_out[:, :, n_order_array]  # todo 输入特征维度

    # 标签转换
    if label_flag < 0:
        label_flag = Y_.min()
    Y_out = trans_label(Y_, label_flag)

    return X_out, Y_out, label_flag
