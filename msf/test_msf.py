#!/usr/bin/env python3
"""
在 data/data_msf 上逐样本 CNN 推理，输出与 stairfit_results.xlsx 类似的 Summary 表。

测试规则（模型 ↔ 数据必须一一对应）：
  - res_500_70w  只测文件名以 500_ 开头的 txt（60 条）
  - res_600_70w  只测 600_* … 以此类推至 1000
  - res_mix_70w  测 500+600+700+800+900+1000 全部（360 条）

输出：
  results/res_500_70w/test_msf/cnn_results.xlsx
  results/res_mix_70w/test_msf/cnn_results.xlsx
  （mix 另含按 Dataset MU 分组的子表 Summary_500 … Summary_1000）

示例：
  python3 msf/test_msf.py
  python3 msf/test_msf.py --only 500 mix
  python3 msf/test_msf.py --dry-run
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import openpyxl as xl
import torch

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "stairfit"))
from StairFit import style_sheet  # noqa: E402
sys.path.insert(0, str(ROOT))
from model import ResNet
from tools import (
    _preprocess_one_sample_1d,
    inverse_label_standardize,
    load_label_norm_npz,
    reverse_label,
)

MSF_DIR = ROOT / "data" / "data_msf"
DATASET_MUS = (500, 600, 700, 800, 900, 1000)
EXP_NAME = "ResCNN_test"
TRANS_FLAG = 5
MAX_MU = 160
MIN_MU = 5
N_ORDER = np.array([0, 1, 2], dtype=np.int64)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


@dataclass(frozen=True)
class TestJob:
    """一条测试任务：用 savedir 下的权重，只处理 files 列表中的 txt。"""

    name: str
    savedir: str
    files: tuple[Path, ...]
    output_dir: Path

    @property
    def output_xlsx(self) -> Path:
        return self.output_dir / "cnn_results.xlsx"


def parse_msf_stem(stem: str) -> tuple[int, float]:
    """500_70w_50_225000_1 -> (dataset_mu=500, label_mu=50)."""
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"无法解析文件名: {stem}")
    return int(parts[0]), float(parts[2])


def load_mscanfit_txt(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size % 2 != 0:
            raise ValueError(f"列数不是偶数: {path}")
        arr = arr.reshape(-1, 2)
    if arr.shape[1] != 2:
        raise ValueError(f"期望两列，当前 {arr.shape}: {path}")
    return arr.T.astype(np.float32)


def waveform_to_tensor(xy: np.ndarray) -> torch.Tensor:
    y = np.asarray(xy[1], dtype=np.float32)
    li = y.size
    if li >= 10:
        if y[:10].sum() > y[-10:].sum():
            y = np.flip(y)
    elif li >= 2 and y[0] > y[-1]:
        y = np.flip(y)
    m = float(np.max(y))
    if m <= 0:
        m = 1.0
    y = y / m
    feats = _preprocess_one_sample_1d(y, N_ORDER)
    x = np.transpose(feats, (1, 0)).astype(np.float32)
    return torch.from_numpy(x).unsqueeze(0)


def load_cnn_model(savedir: str, epoch: int) -> tuple[ResNet, tuple[float, float] | None]:
    fold_dir = ROOT / "results" / savedir / EXP_NAME / "fold0"
    model_path = fold_dir / f"model_epoch{epoch:03d}.pth"
    if not model_path.is_file():
        raise FileNotFoundError(f"未找到权重: {model_path}")
    m_norm, s_norm, en_norm = load_label_norm_npz(fold_dir / "label_norm.npz")
    label_norm = (m_norm, s_norm) if en_norm else None
    model = ResNet(input_size=len(N_ORDER), num_class=1).to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()
    return model, label_norm


def predict_mu(
    model: ResNet,
    x_t: torch.Tensor,
    label_norm: tuple[float, float] | None,
) -> float:
    with torch.no_grad():
        out = model(x_t.to(device))
    pred = out.detach().squeeze().cpu().numpy()
    if label_norm is not None:
        mean, std = label_norm
        pred = inverse_label_standardize(np.asarray(pred), mean, std)
        pred = float(np.round(pred))
    else:
        pred = float(np.round(pred))
    pred = float(reverse_label(np.asarray([pred]), TRANS_FLAG)[0])
    return float(np.clip(pred, MIN_MU, MAX_MU))


def collect_txt_by_dataset_mu() -> dict[int, list[Path]]:
    if not MSF_DIR.is_dir():
        raise FileNotFoundError(f"未找到目录: {MSF_DIR}")
    groups: dict[int, list[Path]] = {mu: [] for mu in DATASET_MUS}
    for p in sorted(MSF_DIR.glob("*.txt")):
        mu, _ = parse_msf_stem(p.stem)
        if mu not in groups:
            raise ValueError(f"未知数据集前缀 MU={mu}: {p.name}")
        groups[mu].append(p)
    return groups


def files_for_mu(groups: dict[int, list[Path]], mu: int) -> list[Path]:
    """res_{mu}_70w 只取文件名前缀为 {mu}_ 的样本。"""
    files = groups[mu]
    bad = [p.name for p in files if parse_msf_stem(p.stem)[0] != mu]
    if bad:
        raise ValueError(f"MU={mu} 组内出现前缀不匹配文件: {bad[:3]}")
    return files


def build_jobs(groups: dict[int, list[Path]]) -> list[TestJob]:
    jobs: list[TestJob] = []
    for mu in DATASET_MUS:
        jobs.append(
            TestJob(
                name=str(mu),
                savedir=f"res_{mu}_70w",
                files=tuple(files_for_mu(groups, mu)),
                output_dir=ROOT / "results" / f"res_{mu}_70w" / "test_msf",
            )
        )
    mix_files: list[Path] = []
    for mu in DATASET_MUS:
        mix_files.extend(files_for_mu(groups, mu))
    jobs.append(
        TestJob(
            name="mix",
            savedir="res_mix_70w",
            files=tuple(mix_files),
            output_dir=ROOT / "results" / "res_mix_70w" / "test_msf",
        )
    )
    return jobs


def print_job_plan(jobs: list[TestJob]) -> None:
    print("测试计划（模型 → 数据前缀）:", flush=True)
    for job in jobs:
        prefixes = sorted({parse_msf_stem(p.stem)[0] for p in job.files})
        print(
            f"  {job.name:>4}  {job.savedir:<16}  N={len(job.files):>3}  "
            f"前缀={prefixes}  -> {job.output_xlsx}",
            flush=True,
        )


def infer_rows(job: TestJob, epoch: int) -> list[dict[str, object]]:
    model, label_norm = load_cnn_model(job.savedir, epoch)
    results: list[dict[str, object]] = []
    for i, path in enumerate(job.files, start=1):
        xy = load_mscanfit_txt(path)
        dataset_mu, true_mu = parse_msf_stem(path.stem)
        predicted = predict_mu(model, waveform_to_tensor(xy), label_norm)
        results.append(
            {
                "file": path.name,
                "rows": int(xy.shape[1]),
                "dataset_mu": dataset_mu,
                "true_mu": true_mu,
                "predicted_mu": predicted,
                "abs_err": abs(predicted - true_mu),
                "epoch": epoch,
                "path": str(path.resolve()),
            }
        )
        if i % 20 == 0 or i == len(job.files):
            print(f"    [{i}/{len(job.files)}] {path.name} pred={predicted:g} true={true_mu:g}", flush=True)
    return results


def _append_summary_sheet(ws, results: list[dict[str, object]]) -> None:
    ws.append(
        [
            "File",
            "Rows",
            "True MU",
            "Predicted MU",
            "Abs Error",
            "Dataset MU",
            "Epoch",
            "Full Path",
        ]
    )
    for r in results:
        ws.append(
            [
                r["file"],
                r["rows"],
                r["true_mu"],
                r["predicted_mu"],
                r["abs_err"],
                r["dataset_mu"],
                r["epoch"],
                r["path"],
            ]
        )
    style_sheet(ws)


def save_cnn_results_excel(job: TestJob, results: list[dict[str, object]]) -> Path:
    wb = xl.Workbook()
    ws_all = wb.active
    ws_all.title = "Summary"
    _append_summary_sheet(ws_all, results)

    if job.name == "mix":
        by_mu: dict[int, list[dict[str, object]]] = {mu: [] for mu in DATASET_MUS}
        for r in results:
            by_mu[int(r["dataset_mu"])].append(r)
        for mu in DATASET_MUS:
            ws = wb.create_sheet(f"Summary_{mu}")
            _append_summary_sheet(ws, by_mu[mu])

    job.output_dir.mkdir(parents=True, exist_ok=True)
    out = job.output_xlsx
    wb.save(str(out))
    return out


def run_job(job: TestJob, epoch: int) -> Path:
    if not job.files:
        raise ValueError(f"任务 {job.name}: 无 txt")
    print(f"\n[job {job.name}] {job.savedir} ← {len(job.files)} 个样本", flush=True)
    t0 = perf_counter()
    results = infer_rows(job, epoch)
    out = save_cnn_results_excel(job, results)
    print(f"  saved {out} ({perf_counter() - t0:.1f}s)", flush=True)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="data_msf：单档模型测单档数据，mix 测全部。")
    p.add_argument(
        "--only",
        nargs="+",
        choices=[*(str(m) for m in DATASET_MUS), "mix", "all"],
        help="只跑指定任务",
    )
    p.add_argument("--epoch", type=int, default=99)
    p.add_argument("--dry-run", action="store_true", help="只打印任务计划，不推理")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    groups = collect_txt_by_dataset_mu()
    jobs = build_jobs(groups)
    print_job_plan(jobs)

    if args.only:
        allow = set(args.only)
        if "all" not in allow:
            jobs = [j for j in jobs if j.name in allow]

    if args.dry_run:
        return

    for job in jobs:
        run_job(job, args.epoch)


if __name__ == "__main__":
    main()
