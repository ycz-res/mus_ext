import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
import time
import re
import numpy as np
from pathlib import Path
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import openpyxl as xl
import matplotlib.pyplot as plt
from model import ResNet
# from half_transformer2 import make_model
from tools import (
    load_data,
    reverse_label,
    load_real_data,
    set_seed,
    VarLenSeqDataset,
    collate_varlen_cnn,
    collate_varlen_cnn_fake_aux,
    fit_label_standardize,
    apply_label_standardize,
    inverse_label_standardize,
    save_label_norm_npz,
    load_label_norm_npz,
)
from torch.utils.tensorboard import SummaryWriter

## 随机种子
# set_seed(8)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False  # 不进行基准测试

n_fold = 4  # todo
batch_size = 256  # todo
num_workers = 0
num_epochs = 100  # todo
mode = 'train'  # train test visual
model_select = 'ResCNN'  # fixed: only ResNet is used
loss_type = 'MSELoss'  # todo MSELoss CrossEntropy
use_label_norm = True  # True：仅对 MSELoss 在训练集上 fit 标准化标签，验证/测试用同一 mean/std
test_trans_flag = 5  # todo 测试集标签最小值
F16 = False  # todo
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
max_mu = 160  # todo 最大MU值
min_mu = 5  # todo 最小MU值


## 定义
def val_in_train(model, loader, label_norm_stats=None):
    res = []
    gold = []
    model.eval()
    start_time = time.time()
    for i, (x_t, y_t) in enumerate(loader):
        with torch.no_grad():
            out = model(x_t.to(device))
        if loss_type == 'CrossEntropy':
            predicted = torch.argmax(out, dim=1)
        else:
            if label_norm_stats is not None:
                predicted = out.detach().squeeze().cpu().numpy()
            else:
                predicted = torch.round(out.data).squeeze().cpu().numpy()
        res.append(predicted)
        gold.append(y_t.cpu().numpy())
    end_time = time.time()
    res = np.concatenate(res, axis=-1)
    gold = np.concatenate(gold, axis=-1)
    if label_norm_stats is not None:
        mean, std = label_norm_stats
        res = inverse_label_standardize(res, mean, std)
        gold = inverse_label_standardize(gold, mean, std)
        res = np.round(res)
        gold = np.round(gold)
    res = reverse_label(res, trans_flag)
    gold = reverse_label(gold, trans_flag)
    res[res > max_mu] = max_mu
    res[res < min_mu] = min_mu
    loss = abs(res - gold).mean()
    correct = (res == gold).sum().item() / len(gold)
    run_time = end_time - start_time
    return loss, correct, run_time


# 注册梯度钩子
def register_gradient_hooks(model, writer_, step):
    for name, param in model.named_parameters():
        def hook(grad, name=name):  # 捕获梯度
            writer_.add_histogram(f"Gradients/{name}", grad, step)

        param.register_hook(hook)


def log_gradient_norms(model, writer_, step):
    total_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)  # L2范数
            total_norm += param_norm.item() ** 2
            writer_.add_scalar(f"GradientNorm/{name}", param_norm, step)
    total_norm = total_norm ** 0.5
    writer_.add_scalar("GradientNorm/Total", total_norm, step)


def train(model, loader_train, loader_val, train_dir, epoch_start=0, label_norm_stats=None):
    # 定义模型、优化器、损失函数和训练参数
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.95), weight_decay=1e-3)  # todo
    if loss_type == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()  # todo 绝对误差损失
    else:
        criterion = nn.MSELoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)  # todo
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(str(train_dir / 'logs' / 'gradient_logs'))

    # 模型训练
    total_batches = len(loader_train)
    for epoch in range(epoch_start, num_epochs):
        gold_list = np.array([])
        prediction_list = np.array([])
        start_time = time.time()
        model.train()
        print(f'[Train] Epoch {epoch + 1}/{num_epochs} start')
        for batch_i, (batch_x, batch_y) in enumerate(loader_train, start=1):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            if F16:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x).squeeze()
                    loss = criterion(outputs, batch_y)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                # loss = (((outputs - batch_y) / (batch_y + 1)) ** 2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            if loss_type == 'CrossEntropy':
                predicted = torch.argmax(outputs, dim=1)
            else:
                if label_norm_stats is not None:
                    predicted = outputs.detach().squeeze().cpu().numpy()
                else:
                    predicted = torch.round(outputs.data).squeeze().cpu().numpy()
            prediction_list = np.concatenate((prediction_list, predicted))
            gold_list = np.concatenate((gold_list, batch_y.cpu().numpy()))
            if batch_i % 10 == 0 or batch_i == total_batches:
                print(f'[Train] Epoch {epoch + 1}/{num_epochs} batch {batch_i}/{total_batches}, loss {loss.item():.4f}')
        scheduler.step()
        end_time = time.time()

        # register_gradient_hooks(model, writer, epoch)
        # log_gradient_norms(model, writer, epoch)  # 记录梯度幅值

        if label_norm_stats is not None:
            mean, std = label_norm_stats
            prediction_list = inverse_label_standardize(prediction_list, mean, std)
            gold_list = inverse_label_standardize(gold_list, mean, std)
            prediction_list = np.round(prediction_list)
            gold_list = np.round(gold_list)
        prediction_list = reverse_label(prediction_list, trans_flag)
        gold_list = reverse_label(gold_list, trans_flag)
        prediction_list[prediction_list > max_mu] = max_mu
        prediction_list[prediction_list < min_mu] = min_mu
        loss_train = abs(prediction_list - gold_list).mean()

        # save model
        torch.save(model.state_dict(), str(train_dir / f'model_epoch{epoch:03d}.pth'))

        # validation
        loss_val, acc_val, run_time = val_in_train(model, loader_val, label_norm_stats=label_norm_stats)
        print(f'Epoch {epoch + 1}/{num_epochs}, train_loss {loss_train:.2f}, val_loss {loss_val:.2f}, '
              f'Time {end_time - start_time:.2f}s')

        # save accuracy
        checkout = Path(train_dir / 'checkout.xlsx')
        if not checkout.exists():
            workbook = xl.Workbook()
            worksheet = workbook.active
            worksheet.append(['epoch', 'loss-train', 'loss-val'])
        else:
            workbook = xl.load_workbook(checkout)
            worksheet = workbook.active
        worksheet.append([epoch, loss_train, loss_val])
        workbook.save(checkout)


## 测试
def test(model, loader, model_path, test_excel, label_norm_stats=None):
    model.load_state_dict(torch.load(model_path))
    prediction, gold, noise, amp, thr, thr_var, hp = [], [], [], [], [], [], []
    model.eval()
    for _, (x_t, y_t, z_t, amp_t, thr_t, thr_var_t, hp_t) in enumerate(loader):
        with torch.no_grad():
            out = model(x_t.to(device))
        if loss_type == 'CrossEntropy':
            predicted = torch.argmax(out, dim=1)
        else:
            if label_norm_stats is not None:
                predicted = out.detach().squeeze().cpu().numpy()
            else:
                predicted = torch.round(out.data).squeeze().cpu().numpy()
        prediction.append(predicted)
        gold.append(y_t.cpu().numpy())
        noise.append(z_t.numpy())
        amp.append(amp_t.numpy())
        thr.append(thr_t.numpy())
        thr_var.append(thr_var_t.numpy())
        hp.append(hp_t.numpy())

    prediction = np.concatenate(prediction, axis=-1)
    gold = np.concatenate(gold, axis=-1)
    noise = np.concatenate(noise, axis=-1)
    amp = np.concatenate(amp, axis=-1)
    thr = np.concatenate(thr, axis=-1)
    thr_var = np.concatenate(thr_var, axis=-1)
    hp = np.concatenate(hp, axis=-1)

    if label_norm_stats is not None and loss_type != 'CrossEntropy':
        mean, std = label_norm_stats
        prediction = inverse_label_standardize(prediction, mean, std)
        gold = inverse_label_standardize(gold, mean, std)
        prediction = np.round(prediction)
        gold = np.round(gold)
    prediction = reverse_label(prediction, trans_flag)
    gold = reverse_label(gold, trans_flag)
    prediction[prediction > max_mu] = max_mu
    prediction[prediction < min_mu] = min_mu

    MAE = abs(prediction - gold).mean()
    MRE = (abs(prediction - gold) / gold).mean()
    print(f'Test: MAE {MAE}, MRE {MRE}')
    xl_out = np.stack((gold, prediction, abs(prediction - gold), noise, amp, thr, thr_var, hp), axis=-1)
    wb = xl.Workbook()
    ws = wb.active
    ws.append(['No', 'gold', 'prediction', 'error', 'noise', 'amp', 'thr', 'thr_var', 'hp', 'MAE', 'MRE'])
    [ws.append([k] + tmp.tolist()) for k, tmp in enumerate(xl_out)]
    ws.cell(2, 10).value = MAE
    ws.cell(2, 11).value = MRE
    wb.save(str(test_excel))
    return xl_out, MAE, MRE


def test_real_data(model, loader, model_path, test_dir_, label_norm_stats=None):
    model.load_state_dict(torch.load(model_path))
    prediction, gold = [], []
    model.eval()
    for _, (x_t, y_t) in enumerate(loader):
        with torch.no_grad():
            out = model(x_t.to(device))
        if loss_type == 'CrossEntropy':
            predicted = torch.argmax(out, dim=1)
        else:
            if label_norm_stats is not None:
                predicted = out.detach().squeeze().cpu().numpy()
            else:
                predicted = torch.round(out.data).squeeze().cpu().numpy()
        prediction.append(predicted)
        gold.append(y_t.cpu().numpy())

    prediction = np.concatenate(prediction, axis=-1)
    gold = np.concatenate(gold, axis=-1)

    if label_norm_stats is not None and loss_type != 'CrossEntropy':
        mean, std = label_norm_stats
        prediction = inverse_label_standardize(prediction, mean, std)
        gold = inverse_label_standardize(gold, mean, std)
        prediction = np.round(prediction)
        gold = np.round(gold)
    prediction = reverse_label(prediction, trans_flag)
    gold = reverse_label(gold, trans_flag)
    prediction[prediction > max_mu] = max_mu
    prediction[prediction < min_mu] = min_mu

    MAE = abs(prediction - gold).mean()
    MRE = (abs(prediction - gold) / gold).mean()
    print(f'Test: MAE {MAE}, MRE {MRE}')
    xl_out = np.stack((gold, prediction, abs(prediction - gold)), axis=-1)
    wb = xl.Workbook()
    ws = wb.active
    ws.append(['gold', 'prediction', 'error', 'MAE', 'MRE'])
    [ws.append(tmp.tolist()) for tmp in xl_out]
    ws.cell(2, 6).value = MAE
    ws.cell(2, 6).value = MRE
    if test_dir_.is_dir():
        wb.save(str(test_dir_ / 'test.xlsx'))
    else:
        wb.save(str(test_dir_))
    return xl_out, MAE, MRE


## load data
Exp_name = 'ResCNN_test'  # todo 指定实验名称
assert model_select in Exp_name
root_dir = Path("./").resolve()
save_dir = root_dir / 'results' / 'MUNE_simple' / Exp_name
if not save_dir.exists():
    save_dir.mkdir(parents=True, exist_ok=True)

if mode == 'train':
    train_dataset_name = ['train_mix_40w.mat']  # train_500_20w + train_1000_20w 混合
    val_dataset_name = ['dev_mix_32k.mat']  # dev_500_1.6w + dev_1000_1.6w 混合
    train_model = None  # save_dir / 'fold1' / 'model_epoch59.pth'  # todo 指定训练模型保存路径
    epoch_resume = 0  # todo

    for kf_i in range(len(train_dataset_name)):
        print(f'===== Fold {kf_i + 1}/{len(train_dataset_name)} =====')
        set_seed(kf_i)  # todo 设置随机种子
        # 加载训练数据
        X_train, Y_train, n_class, x_len, x_dim, trans_flag, train_varlen = load_data(
            root_dir / 'data' / train_dataset_name[kf_i], model_select, -1, loss_type)
        print(f'训练集样本数：{len(X_train)}, trans_flag: {trans_flag}, varlen: {train_varlen}')
        # 加载验证数据
        X_val, Y_val, _, _, _, _, val_varlen = load_data(
            root_dir / 'data' / val_dataset_name[kf_i], model_select, trans_flag, loss_type)
        print(f'验证集样本数：{len(X_val)}, trans_flag: {trans_flag}, varlen: {val_varlen}')
        label_norm_stats = None
        if use_label_norm and loss_type == 'MSELoss':
            mean, std = fit_label_standardize(np.asarray(Y_train))
            Y_train = apply_label_standardize(np.asarray(Y_train), mean, std)
            Y_val = apply_label_standardize(np.asarray(Y_val), mean, std)
            label_norm_stats = (mean, std)
            print(f'标签标准化: mean={mean:.4f}, std={std:.4f}（仅由训练集估计）')
        if train_varlen:
            train_dataset = VarLenSeqDataset(X_train, Y_train)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_varlen_cnn,
            )
        else:
            trainX, trainY = torch.tensor(X_train), torch.tensor(Y_train)
            train_dataset = TensorDataset(trainX, trainY)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=True)

        if val_varlen:
            val_dataset = VarLenSeqDataset(X_val, Y_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_varlen_cnn,
            )
        else:
            valX, valY = torch.tensor(X_val), torch.tensor(Y_val)
            val_dataset = TensorDataset(valX, valY)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                    num_workers=num_workers, pin_memory=True)

        model_i = ResNet(input_size=x_dim, num_class=n_class).to(device)  # 创建模型
        if train_model is None:
            kf_dir = save_dir / f'fold{kf_i}'
            kf_dir.mkdir() if not kf_dir.exists() else None
        else:  # 加载训练好的模型
            kf_dir = train_model.parent
            model_i.load_state_dict(torch.load(str(train_model)))

        if label_norm_stats is not None:
            save_label_norm_npz(kf_dir / 'label_norm.npz', label_norm_stats[0], label_norm_stats[1], True)
        else:
            save_label_norm_npz(kf_dir / 'label_norm.npz', 0.0, 1.0, False)

        # 训练
        train(model_i, train_loader, val_loader, kf_dir, epoch_resume, label_norm_stats=label_norm_stats)

if mode == 'test':
    data_type = 'fake'  # todo  fake or real
    test_dataset_name = 'dev_mix_32k.mat'
    # 'test_dataset_T2_HP_better_range_v2' 'real_data_control' 'real_data_sci' 'test_dataset_T1_HP_better_range_10'
    model_file = save_dir / 'fold0'  # todo 指定模型文件或文件夹；单文件如 fold0/model_epoch099.pth 只测该 checkpoint
    if model_file.is_dir():
        test_dir = model_file
    else:
        test_dir = model_file.parent
    test_dir = test_dir / 'test_results'  # todo 指定测试结果保存路径
    test_dir.mkdir(exist_ok=True)
    if model_file.is_dir():
        # 按照epoch排序
        model_file = list(model_file.glob('*.pth'))
        epoch_list = np.zeros(len(model_file), dtype=int)
        for i, file_i in enumerate(model_file):
            epoch_str = re.search(r'_epoch(\d+)\.pth$', str(file_i))
            epoch_num = int(epoch_str.group(1))
            epoch_list[i] = epoch_num
        index_sort = np.argsort(epoch_list)
        mode_file_sort = [model_file[i] for i in index_sort]
        model_file = mode_file_sort
    else:
        model_file = [model_file]

    norm_path = Path(model_file[0]).resolve().parent / 'label_norm.npz'
    m_norm, s_norm, en_norm = load_label_norm_npz(norm_path)
    label_norm_test = (m_norm, s_norm) if en_norm else None
    if label_norm_test is not None:
        print(f'[test] label_norm: mean={label_norm_test[0]:.4f}, std={label_norm_test[1]:.4f}')

    # build test dataset
    if data_type == 'fake':
        X_test, Y_test, n_class, x_len, x_dim, trans_flag, test_varlen = load_data(
            root_dir / 'data' / test_dataset_name, model_select, test_trans_flag, loss_type)
        print(f'测试集样本数：{len(X_test)}, trans_flag: {trans_flag}, varlen: {test_varlen}')
        if test_varlen and len(Y_test.shape) != 1:
            raise NotImplementedError("变长测试暂只支持单列标签 Y；请使用固定长度 .mat 或多列标签时关闭 seq_len")
        if label_norm_test is not None and len(Y_test.shape) == 1 and loss_type == 'MSELoss':
            Y_test = apply_label_standardize(np.asarray(Y_test), label_norm_test[0], label_norm_test[1])
        if len(Y_test.shape) == 1:
            testY = torch.tensor(Y_test)
            zero_feature = torch.zeros(len(X_test), dtype=torch.float32)
            test_noise, test_amp = zero_feature, zero_feature
            test_thr, test_thr_var = zero_feature, zero_feature
            test_hp = zero_feature
        else:
            testY = torch.tensor(Y_test[:, 0])
            test_noise, test_amp = torch.tensor(Y_test[:, 1]), torch.tensor(Y_test[:, 2])
            test_thr, test_thr_var = torch.tensor(Y_test[:, 3]), torch.tensor(Y_test[:, 4])
            if Y_test.shape[1] >= 6:
                test_hp = torch.tensor(Y_test[:, 5])
            else:
                test_hp = torch.zeros(len(X_test), dtype=torch.float32)
        if test_varlen:
            test_dataset = VarLenSeqDataset(X_test, Y_test)
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_varlen_cnn_fake_aux,
            )
        else:
            testX = torch.tensor(X_test)
            test_dataset = TensorDataset(testX, testY, test_noise, test_amp, test_thr, test_thr_var, test_hp)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                     num_workers=num_workers, pin_memory=True)
    else:
        X_test, Y_test, n_class, x_len, x_dim, trans_flag, test_varlen = load_real_data(
            root_dir / 'data' / test_dataset_name, model_select, test_trans_flag, loss_type)
        if label_norm_test is not None and len(Y_test.shape) == 1 and loss_type == 'MSELoss':
            Y_test = apply_label_standardize(np.asarray(Y_test), label_norm_test[0], label_norm_test[1])
        if test_varlen:
            test_dataset = VarLenSeqDataset(X_test, Y_test)
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_varlen_cnn,
            )
        else:
            testX, testY = torch.tensor(X_test), torch.tensor(Y_test)
            test_dataset = TensorDataset(testX, testY)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                     num_workers=num_workers, pin_memory=True)
    # 模型测试
    test_loss_list = np.zeros((len(model_file), 3))
    for i, model_file_i in enumerate(model_file):
        epoch_test = re.search(r'_epoch(\d+)\.pth$', str(model_file_i))
        epoch_test = epoch_test.group(1)
        test_excel_i = test_dir / ('epoch_' + epoch_test + '.xlsx')
        model_i = ResNet(input_size=x_dim, num_class=n_class).to(device)
        if data_type == 'real':  # 真实数据测试
            test_res, test_MAE, test_MRE = test_real_data(
                model_i, test_loader, str(model_file_i), test_excel_i, label_norm_stats=label_norm_test
            )
        else:  # 仿真数据测试
            test_res, test_MAE, test_MRE = test(
                model_i, test_loader, str(model_file_i), test_excel_i, label_norm_stats=label_norm_test
            )
        test_loss_list[i] = np.array([int(epoch_test), test_MAE, test_MRE])
    workbook_ = xl.Workbook()
    worksheet_ = workbook_.active
    worksheet_.append(['epoch', 'test_MAE', 'test_MRE'])
    [worksheet_.append(tmp.tolist()) for tmp in test_loss_list]
    workbook_.save(str(test_dir / 'test_results.xlsx'))

