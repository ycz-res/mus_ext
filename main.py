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
from tools import load_data, reverse_label, load_real_data, set_seed
from torch.utils.tensorboard import SummaryWriter

## 随机种子
# set_seed(8)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False  # 不进行基准测试

n_fold = 4  # todo
batch_size = 256  # todo
num_workers = 0
num_epochs = 100  # todo
mode = 'test'  # train test visual
model_select = 'ResCNN'  # fixed: only ResNet is used
loss_type = 'MSELoss'  # todo MSELoss CrossEntropy
test_trans_flag = 5  # todo 测试集标签最小值
F16 = False  # todo
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
max_mu = 160  # todo 最大MU值
min_mu = 5  # todo 最小MU值


## 定义
def val_in_train(model, loader):
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
            predicted = torch.round(out.data).squeeze()
        res.append(predicted.cpu().numpy())
        gold.append(y_t.cpu().numpy())
    end_time = time.time()
    res = np.concatenate(res, axis=-1)
    gold = np.concatenate(gold, axis=-1)
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


def train(model, loader_train, loader_val, train_dir, epoch_start=0):
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
                predicted = torch.round(outputs.data).squeeze()
            prediction_list = np.concatenate((prediction_list, predicted.cpu().numpy()))
            gold_list = np.concatenate((gold_list, batch_y.cpu().numpy()))
            if batch_i % 10 == 0 or batch_i == total_batches:
                print(f'[Train] Epoch {epoch + 1}/{num_epochs} batch {batch_i}/{total_batches}, loss {loss.item():.4f}')
        scheduler.step()
        end_time = time.time()

        # register_gradient_hooks(model, writer, epoch)
        # log_gradient_norms(model, writer, epoch)  # 记录梯度幅值

        prediction_list = reverse_label(prediction_list, trans_flag)
        gold_list = reverse_label(gold_list, trans_flag)
        prediction_list[prediction_list > max_mu] = max_mu
        prediction_list[prediction_list < min_mu] = min_mu
        loss_train = abs(prediction_list - gold_list).mean()

        # save model
        torch.save(model.state_dict(), str(train_dir / f'model_epoch{epoch:03d}.pth'))

        # validation
        loss_val, acc_val, run_time = val_in_train(model, loader_val)
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
def test(model, loader, model_path, test_excel):
    model.load_state_dict(torch.load(model_path))
    prediction, gold, noise, amp, thr, thr_var, hp = [], [], [], [], [], [], []
    model.eval()
    for _, (x_t, y_t, z_t, amp_t, thr_t, thr_var_t, hp_t) in enumerate(loader):
        with torch.no_grad():
            out = model(x_t.to(device))
        if loss_type == 'CrossEntropy':
            predicted = torch.argmax(out, dim=1)
        else:
            predicted = torch.round(out.data).squeeze()
        prediction.append(predicted.cpu().numpy())
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


def test_real_data(model, loader, model_path, test_dir_):
    model.load_state_dict(torch.load(model_path))
    prediction, gold = [], []
    model.eval()
    for _, (x_t, y_t) in enumerate(loader):
        with torch.no_grad():
            out = model(x_t.to(device))
        if loss_type == 'CrossEntropy':
            predicted = torch.argmax(out, dim=1)
        else:
            predicted = torch.round(out.data).squeeze()
        prediction.append(predicted.cpu().numpy())
        gold.append(y_t.cpu().numpy())

    prediction = np.concatenate(prediction, axis=-1)
    gold = np.concatenate(gold, axis=-1)

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
    train_dataset_name = ['train_1000_20w.mat',
                          ]  # todo 指定训练集
    val_dataset_name = ['dev_1000_1.6w.mat',
                        ]  # todo 指定验证集
    train_model = None  # save_dir / 'fold1' / 'model_epoch59.pth'  # todo 指定训练模型保存路径
    epoch_resume = 0  # todo

    for kf_i in range(len(train_dataset_name)):
        print(f'===== Fold {kf_i + 1}/{len(train_dataset_name)} =====')
        set_seed(kf_i)  # todo 设置随机种子
        # 加载训练数据
        X_train, Y_train, n_class, x_len, x_dim, trans_flag = load_data(
            root_dir / 'data' / train_dataset_name[kf_i], model_select, -1, loss_type)
        print(f'训练集样本数：{len(X_train)}, trans_flag: {trans_flag}')
        # 加载验证数据
        X_val, Y_val, _, _, _, _ = load_data(
            root_dir / 'data' / val_dataset_name[kf_i], model_select, trans_flag, loss_type)
        print(f'验证集样本数：{len(X_val)}, trans_flag: {trans_flag}')
        trainX, trainY = torch.tensor(X_train), torch.tensor(Y_train)
        train_dataset = TensorDataset(trainX, trainY)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)

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

        # 训练
        train(model_i, train_loader, val_loader, kf_dir, epoch_resume)

if mode == 'test':
    data_type = 'fake'  # todo  fake or real
    test_dataset_name = 'dev_1000_1.6w.mat'  # todo 指定测试集
    # 'test_dataset_T2_HP_better_range_v2' 'real_data_control' 'real_data_sci' 'test_dataset_T1_HP_better_range_10'
    model_file = save_dir / 'fold0'  # todo 指定模型文件或文件夹  model_epoch20.pth
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

    # build test dataset
    if data_type == 'fake':
        X_test, Y_test, n_class, x_len, x_dim, trans_flag = load_data(
            root_dir / 'data' / test_dataset_name, model_select, test_trans_flag, loss_type)
        print(f'测试集样本数：{len(X_test)}, trans_flag: {trans_flag}')
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
        testX = torch.tensor(X_test)
        test_dataset = TensorDataset(testX, testY, test_noise, test_amp, test_thr, test_thr_var, test_hp)
    else:
        X_test, Y_test, n_class, x_len, x_dim, trans_flag = load_real_data(
            root_dir / 'data' / test_dataset_name, model_select, test_trans_flag, loss_type)
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
            test_res, test_MAE, test_MRE = test_real_data(model_i, test_loader, str(model_file_i), test_excel_i)
        else:  # 仿真数据测试
            test_res, test_MAE, test_MRE = test(model_i, test_loader, str(model_file_i), test_excel_i)
        test_loss_list[i] = np.array([int(epoch_test), test_MAE, test_MRE])
    workbook_ = xl.Workbook()
    worksheet_ = workbook_.active
    worksheet_.append(['epoch', 'test_MAE', 'test_MRE'])
    [worksheet_.append(tmp.tolist()) for tmp in test_loss_list]
    workbook_.save(str(test_dir / 'test_results.xlsx'))

