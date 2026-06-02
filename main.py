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
from model import ResNet, CNNsimple, BiLSTM_GAMP, Transformer_CLS
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
mode = 'train'  # train test test_seg visual
train_dataset_name = ['split_70w/500_70w_train.mat']  # 500/600/700/800/900/1000/mix 换对应路径
val_dataset_name = ['split_70w/500_70w_val.mat']
test_dataset_name = 'split_70w/500_70w_test.mat'
savedir = 'cmp_net/res_500_70w'  # 与档位一致：res_600_70w、res_mix_70w 等
test_outdir = 'test_seg'  # test/test_seg：测试结果保存到 results/<savedir>/...

model_select = 'Transformer_CLS'  # ResCNN | CNNsimple | BiLSTM_GAMP | Transformer_CLS
loss_type = 'MSELoss'  # todo MSELoss CrossEntropy
use_label_norm = True  # True：仅对 MSELoss 在训练集上 fit 标准化标签，验证/测试用同一 mean/std
test_trans_flag = 5  # todo 测试集标签最小值
F16 = False  # todo
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
max_mu = 160  # todo 最大MU值
min_mu = 5  # todo 最小MU值


def compute_steprx_d50(prediction, gold, eps=1e-6):
    """
    steprx: 相对误差均方根 RMSRE = sqrt(mean(((pred - gold) / gold)^2))，与 MRE 互补。
    d50: 绝对误差的 50% 分位数（中位绝对误差）。
    """
    g = np.maximum(np.asarray(gold, dtype=np.float64), eps)
    rel = (np.asarray(prediction, dtype=np.float64) - g) / g
    steprx = float(np.sqrt(np.mean(rel ** 2)))
    d50 = float(np.percentile(np.abs(prediction - gold), 50))
    return steprx, d50


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
    steprx, d50 = compute_steprx_d50(prediction, gold)
    print(f'Test: MAE {MAE}, MRE {MRE}, steprx {steprx}, d50 {d50}')
    xl_out = np.stack((gold, prediction, abs(prediction - gold), noise, amp, thr, thr_var, hp), axis=-1)
    wb = xl.Workbook()
    ws = wb.active
    ws.append(
        ['No', 'gold', 'prediction', 'error', 'noise', 'amp', 'thr', 'thr_var', 'hp', 'MAE', 'MRE', 'steprx', 'd50']
    )
    [ws.append([k] + tmp.tolist()) for k, tmp in enumerate(xl_out)]
    ws.cell(2, 10).value = MAE
    ws.cell(2, 11).value = MRE
    ws.cell(2, 12).value = steprx
    ws.cell(2, 13).value = d50
    wb.save(str(test_excel))
    return xl_out, MAE, MRE, steprx, d50



def _format_model_output(out, label_norm_stats=None):
    """将模型输出转成 numpy prediction，保持与 test()/test_real_data() 一致。"""
    if loss_type == 'CrossEntropy':
        predicted = torch.argmax(out, dim=1).detach().cpu().numpy()
    else:
        if label_norm_stats is not None:
            predicted = out.detach().squeeze().cpu().numpy()
        else:
            predicted = torch.round(out.data).squeeze().cpu().numpy()
    return np.atleast_1d(predicted)


def _finalize_prediction_metrics(prediction_list, gold_list, label_norm_stats=None):
    """拼接、反标准化、reverse_label、clip，并计算 MAE/MRE/steprx/d50。"""
    prediction = np.concatenate(prediction_list, axis=-1)
    gold = np.concatenate(gold_list, axis=-1)

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

    MAE = float(abs(prediction - gold).mean())
    MRE = float((abs(prediction - gold) / gold).mean())
    steprx, d50 = compute_steprx_d50(prediction, gold)
    return prediction, gold, MAE, MRE, steprx, d50


def _run_eval_pass(model, loader, label_norm_stats=None, mask_range=None):
    """
    跑一遍测试集。
    mask_range=None 表示 baseline；否则 mask_range=(start, end)，对输入最后一维 L 做置零。
    兼容 fake loader 的 7 元组和 real loader 的 2 元组。
    """
    prediction, gold = [], []
    model.eval()
    for batch in loader:
        x_t, y_t = batch[0], batch[1]
        if mask_range is not None:
            start, end = mask_range
            x_in = x_t.clone()
            # x_in shape: (B, C, L)，刺激点位置在最后一维 L。
            cur_l = x_in.shape[-1]
            start_i = max(0, min(int(start), cur_l))
            end_i = max(0, min(int(end), cur_l))
            if start_i < end_i:
                x_in[:, :, start_i:end_i] = 0
        else:
            x_in = x_t

        with torch.no_grad():
            out = model(x_in.to(device))
        predicted = _format_model_output(out, label_norm_stats=label_norm_stats)
        prediction.append(predicted)
        gold.append(np.atleast_1d(y_t.cpu().numpy()))

    return _finalize_prediction_metrics(prediction, gold, label_norm_stats=label_norm_stats)


def _infer_loader_max_len(loader):
    """推断当前测试 loader 中输入序列最后一维的最大长度。"""
    max_l = 0
    for batch in loader:
        x_t = batch[0]
        max_l = max(max_l, int(x_t.shape[-1]))
    if max_l <= 0:
        raise ValueError('无法从 test loader 推断输入长度，请检查数据。')
    return max_l


def _save_detail_xlsx(detail_excel, sheet_name, header, rows):
    """
    保存样本级明细到 xlsx。
    注意：当样本数很大、segment 很多时，xlsx 会比较大，但便于直接查看和论文整理。
    """
    detail_excel = Path(detail_excel)
    detail_excel.parent.mkdir(parents=True, exist_ok=True)

    wb = xl.Workbook()
    ws = wb.active
    ws.title = sheet_name
    ws.append(header)

    for row in rows:
        ws.append(row)

    wb.save(str(detail_excel))


def test_seg_effect(
        model,
        loader,
        model_path,
        test_excel,
        label_norm_stats=None,
        segment_size=50,
        max_len=None,
        top_k=5,
        save_detail=True,
):
    """
    测试阶段刺激段影响实验：正常训练好的模型不变，只在 test 时逐段 mask 输入。

    目的：找到模型最适配/最依赖的刺激点段。
    - 默认 max_len=None：分析当前测试输入的完整长度，例如 500->10段，1000->20段。
    - 若 max_len=500：只分析共同前500维，适合跨维度共同区域对比。
    - 输入 x 的形状应为 (B, C, L)，mask 发生在最后一维 L 上。

    输出文件：
    1) epoch_XXX.xlsx
       - baseline
       - seg_effect
       - top_by_delta_MAE
    2) epoch_XXX_seg_delta.npy
       - shape=(num_segments, 4)
       - columns=[delta_MAE, delta_MRE, delta_steprx, delta_d50]
    3) epoch_XXX_seg_delta.xlsx
       - seg_delta 的 Excel 表格版
    4) epoch_XXX_seg_metrics.npy
       - shape=(num_segments, 4)
       - columns=[mask_MAE, mask_MRE, mask_steprx, mask_d50]
    5) epoch_XXX_seg_metrics.xlsx
       - seg_metrics 的 Excel 表格版
    6) epoch_XXX_baseline_detail.xlsx
       - baseline 每个样本的 gold / prediction / error
    7) epoch_XXX_seg_detail.xlsx
       - 每个 segment mask 后，每个样本的 gold / prediction / error
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if segment_size <= 0:
        raise ValueError(f'segment_size 必须为正数，当前为 {segment_size}')

    input_max_len = _infer_loader_max_len(loader)
    analysis_len = input_max_len if max_len is None else min(int(max_len), input_max_len)
    num_segments = int(np.ceil(analysis_len / segment_size))
    test_excel = Path(test_excel)
    test_excel.parent.mkdir(parents=True, exist_ok=True)

    # 1) baseline：完整输入测试
    baseline_pred, baseline_gold, base_MAE, base_MRE, base_steprx, base_d50 = _run_eval_pass(
        model, loader, label_norm_stats=label_norm_stats, mask_range=None
    )

    baseline_error = np.abs(baseline_pred - baseline_gold)

    print(
        f'Baseline: MAE {base_MAE}, MRE {base_MRE}, '
        f'steprx {base_steprx}, d50 {base_d50}, input_len {input_max_len}, analysis_len {analysis_len}'
    )

    # baseline 样本级明细
    baseline_detail_rows = []
    for sample_i in range(len(baseline_gold)):
        baseline_detail_rows.append([
            sample_i,
            float(baseline_gold[sample_i]),
            float(baseline_pred[sample_i]),
            float(baseline_error[sample_i]),
        ])

    # 2) 逐段 mask，计算每个刺激段的误差增量，并保存每个样本的预测结果
    rows = []
    detail_rows = []
    delta_array = np.zeros((num_segments, 4), dtype=np.float64)
    metric_array = np.zeros((num_segments, 4), dtype=np.float64)

    for seg_i in range(num_segments):
        start = seg_i * segment_size
        end = min((seg_i + 1) * segment_size, analysis_len)

        seg_pred, seg_gold, MAE_i, MRE_i, steprx_i, d50_i = _run_eval_pass(
            model, loader, label_norm_stats=label_norm_stats, mask_range=(start, end)
        )

        seg_error = np.abs(seg_pred - seg_gold)

        d_MAE = MAE_i - base_MAE
        d_MRE = MRE_i - base_MRE
        d_steprx = steprx_i - base_steprx
        d_d50 = d50_i - base_d50

        metric_array[seg_i] = np.array([MAE_i, MRE_i, steprx_i, d50_i])
        delta_array[seg_i] = np.array([d_MAE, d_MRE, d_steprx, d_d50])

        rows.append([
            seg_i,
            f'S{seg_i + 1}',
            start,
            end - 1,
            MAE_i,
            MRE_i,
            steprx_i,
            d50_i,
            d_MAE,
            d_MRE,
            d_steprx,
            d_d50,
        ])

        # segment 样本级明细
        for sample_i in range(len(seg_gold)):
            detail_rows.append([
                seg_i,
                f'S{seg_i + 1}',
                start,
                end - 1,
                sample_i,
                float(seg_gold[sample_i]),
                float(seg_pred[sample_i]),
                float(seg_error[sample_i]),
            ])

        print(
            f'Segment {seg_i} [{start}:{end}): '
            f'ΔMAE={d_MAE:.6f}, ΔMRE={d_MRE:.6f}, '
            f'Δsteprx={d_steprx:.6f}, Δd50={d_d50:.6f}'
        )

    # 3) 保存 npy，便于后续画热力图/跨维度汇总
    delta_npy = test_excel.with_name(test_excel.stem + '_seg_delta.npy')
    metrics_npy = test_excel.with_name(test_excel.stem + '_seg_metrics.npy')
    np.save(delta_npy, delta_array)
    np.save(metrics_npy, metric_array)

    # 额外保存 npy 对应的 Excel 表格版，便于直接查看和汇总
    delta_excel = test_excel.with_name(test_excel.stem + '_seg_delta.xlsx')
    wb_delta = xl.Workbook()
    ws_delta = wb_delta.active
    ws_delta.title = 'seg_delta'
    ws_delta.append([
        'segment_index',
        'segment_name',
        'start',
        'end',
        'delta_MAE',
        'delta_MRE',
        'delta_steprx',
        'delta_d50'
    ])
    for seg_i in range(num_segments):
        start = seg_i * segment_size
        end = min((seg_i + 1) * segment_size, analysis_len)
        ws_delta.append([
            seg_i,
            f'S{seg_i + 1}',
            start,
            end - 1,
            float(delta_array[seg_i, 0]),
            float(delta_array[seg_i, 1]),
            float(delta_array[seg_i, 2]),
            float(delta_array[seg_i, 3]),
        ])
    wb_delta.save(str(delta_excel))

    metrics_excel = test_excel.with_name(test_excel.stem + '_seg_metrics.xlsx')
    wb_metrics = xl.Workbook()
    ws_metrics = wb_metrics.active
    ws_metrics.title = 'seg_metrics'
    ws_metrics.append([
        'segment_index',
        'segment_name',
        'start',
        'end',
        'mask_MAE',
        'mask_MRE',
        'mask_steprx',
        'mask_d50'
    ])
    for seg_i in range(num_segments):
        start = seg_i * segment_size
        end = min((seg_i + 1) * segment_size, analysis_len)
        ws_metrics.append([
            seg_i,
            f'S{seg_i + 1}',
            start,
            end - 1,
            float(metric_array[seg_i, 0]),
            float(metric_array[seg_i, 1]),
            float(metric_array[seg_i, 2]),
            float(metric_array[seg_i, 3]),
        ])
    wb_metrics.save(str(metrics_excel))

    # 4) 保存样本级明细 xlsx
    baseline_detail_excel = test_excel.with_name(test_excel.stem + '_baseline_detail.xlsx')
    seg_detail_excel = test_excel.with_name(test_excel.stem + '_seg_detail.xlsx')

    if save_detail:
        _save_detail_xlsx(
            baseline_detail_excel,
            'baseline_detail',
            ['No', 'gold', 'prediction', 'error'],
            baseline_detail_rows
        )

        _save_detail_xlsx(
            seg_detail_excel,
            'seg_detail',
            ['segment_index', 'segment_name', 'start', 'end', 'No', 'gold', 'prediction', 'error'],
            detail_rows
        )

    # 5) 保存主 Excel：baseline + 全部分段结果 + TopK
    wb = xl.Workbook()

    ws_base = wb.active
    ws_base.title = 'baseline'
    ws_base.append(['input_len', 'analysis_len', 'segment_size', 'MAE', 'MRE', 'steprx', 'd50'])
    ws_base.append([input_max_len, analysis_len, segment_size, base_MAE, base_MRE, base_steprx, base_d50])

    ws = wb.create_sheet('seg_effect')
    ws.append([
        'segment_index', 'segment_name', 'start', 'end',
        'mask_MAE', 'mask_MRE', 'mask_steprx', 'mask_d50',
        'delta_MAE', 'delta_MRE', 'delta_steprx', 'delta_d50'
    ])
    for row in rows:
        ws.append(row)

    ws_top = wb.create_sheet('top_by_delta_MAE')
    ws_top.append([
        'rank', 'segment_index', 'segment_name', 'start', 'end',
        'delta_MAE', 'delta_MRE', 'delta_steprx', 'delta_d50'
    ])

    order = np.argsort(-delta_array[:, 0])
    for rank, idx in enumerate(order[:min(top_k, len(order))], start=1):
        row = rows[int(idx)]
        ws_top.append([rank, row[0], row[1], row[2], row[3], row[8], row[9], row[10], row[11]])

    wb.save(str(test_excel))

    print(f'[TestSeg] saved excel: {test_excel}')
    print(f'[TestSeg] saved delta npy: {delta_npy}')
    print(f'[TestSeg] saved delta excel: {delta_excel}')
    print(f'[TestSeg] saved metrics npy: {metrics_npy}')
    print(f'[TestSeg] saved metrics excel: {metrics_excel}')
    if save_detail:
        print(f'[TestSeg] saved baseline detail excel: {baseline_detail_excel}')
        print(f'[TestSeg] saved segment detail excel: {seg_detail_excel}')

    # 返回 baseline 预测结果，以及 baseline 和消融结果，兼容后续主流程记录 baseline 指标。
    xl_out = np.stack((baseline_gold, baseline_pred, baseline_error), axis=-1)
    return xl_out, base_MAE, base_MRE, base_steprx, base_d50, delta_array, metric_array

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
    steprx, d50 = compute_steprx_d50(prediction, gold)
    print(f'Test: MAE {MAE}, MRE {MRE}, steprx {steprx}, d50 {d50}')
    xl_out = np.stack((gold, prediction, abs(prediction - gold)), axis=-1)
    wb = xl.Workbook()
    ws = wb.active
    ws.append(['gold', 'prediction', 'error', 'MAE', 'MRE', 'steprx', 'd50'])
    [ws.append(tmp.tolist()) for tmp in xl_out]
    ws.cell(2, 4).value = MAE
    ws.cell(2, 5).value = MRE
    ws.cell(2, 6).value = steprx
    ws.cell(2, 7).value = d50
    if test_dir_.is_dir():
        wb.save(str(test_dir_ / 'test.xlsx'))
    else:
        wb.save(str(test_dir_))
    return xl_out, MAE, MRE, steprx, d50


## load data
Exp_name = f'{model_select}_test'
assert model_select in Exp_name
root_dir = Path("./").resolve()
save_dir = root_dir / 'results' / savedir / Exp_name
if not save_dir.exists():
    save_dir.mkdir(parents=True, exist_ok=True)

if mode == 'train':
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

        if model_select == 'ResCNN':
            model_i = ResNet(input_size=x_dim, num_class=n_class).to(device)
        elif model_select == 'CNNsimple':
            model_i = CNNsimple(input_size=x_dim, num_class=n_class).to(device)
        elif model_select == 'BiLSTM_GAMP':
            model_i = BiLSTM_GAMP(input_size=x_dim, num_class=n_class).to(device)
        elif model_select == 'Transformer_CLS':
            model_i = Transformer_CLS(input_size=x_dim, num_class=n_class).to(device)
        else:
            raise ValueError(f'未知 model_select: {model_select}')
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

if mode in ['test', 'test_seg']:
    data_type = 'fake'  # todo  fake or real
    # 'test_dataset_T2_HP_better_range_v2' 'real_data_control' 'real_data_sci' 'test_dataset_T1_HP_better_range_10'
    model_file = save_dir / 'fold0' / 'model_epoch099.pth'
    test_dir = root_dir / 'results' / savedir / test_outdir
    test_dir.mkdir(parents=True, exist_ok=True)
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
    test_loss_list = np.zeros((len(model_file), 5))
    for i, model_file_i in enumerate(model_file):
        epoch_test = re.search(r'_epoch(\d+)\.pth$', str(model_file_i))
        epoch_test = epoch_test.group(1)
        test_excel_i = test_dir / ('epoch_' + epoch_test + '.xlsx')
        if model_select == 'ResCNN':
            model_i = ResNet(input_size=x_dim, num_class=n_class).to(device)
        elif model_select == 'CNNsimple':
            model_i = CNNsimple(input_size=x_dim, num_class=n_class).to(device)
        elif model_select == 'BiLSTM_GAMP':
            model_i = BiLSTM_GAMP(input_size=x_dim, num_class=n_class).to(device)
        elif model_select == 'Transformer_CLS':
            model_i = Transformer_CLS(input_size=x_dim, num_class=n_class).to(device)
        else:
            raise ValueError(f'未知 model_select: {model_select}')
        if mode == 'test_seg':
            # 完整长度刺激段影响测试：默认 max_len=None，会分析当前输入的完整长度。
            # 若只想比较所有维度共同的前500维，可改成 max_len=500。
            test_res, test_MAE, test_MRE, test_steprx, test_d50, _, _ = test_seg_effect(
                model_i,
                test_loader,
                str(model_file_i),
                test_excel_i,
                label_norm_stats=label_norm_test,
                segment_size=50,
                max_len=None,
                top_k=5,
                save_detail=True,
            )
        elif data_type == 'real':  # 真实数据测试
            test_res, test_MAE, test_MRE, test_steprx, test_d50 = test_real_data(
                model_i, test_loader, str(model_file_i), test_excel_i, label_norm_stats=label_norm_test
            )
        else:  # 仿真数据测试
            test_res, test_MAE, test_MRE, test_steprx, test_d50 = test(
                model_i, test_loader, str(model_file_i), test_excel_i, label_norm_stats=label_norm_test
            )
        test_loss_list[i] = np.array([int(epoch_test), test_MAE, test_MRE, test_steprx, test_d50])
    workbook_ = xl.Workbook()
    worksheet_ = workbook_.active
    worksheet_.append(['epoch', 'test_MAE', 'test_MRE', 'test_steprx', 'test_d50'])
    [worksheet_.append(tmp.tolist()) for tmp in test_loss_list]
    workbook_.save(str(test_dir / 'test_results.xlsx'))

