#!/usr/bin/env python3
"""
从 MATLAB v7.3（HDF5）.mat 导出 MScanFit 可用的双列曲线 .txt。

约定字段：
  - data:      shape (2, L, N)，``data[0, :, i]`` 为 x，``data[1, :, i]`` 为 y
  - label_num: shape (1, N) 或可 squeeze 为一维

输出格式：两列、制表符分隔、CRLF 行尾，与 MScanFit 常见输入一致。

文件名：``<前缀>_<label>_<真实索引>_<序号>.txt``。

- **真实索引**：.mat 样本维列下标 ``j``（从 **0** 起，与 ``data[:, :, j]`` 一致）。
- **序号**：该 label 下第几条导出（从 **1** 起，连续）。

例如 ``500_70w_50_1240_3.txt`` 表示前缀 ``500_70w``、label 50、全局列 ``j=1240``、该 label 下第 3 条导出。

默认输出目录为仓库下的 ``data/data_msf``；可用 ``--out-dir`` 改成其它路径（例如 ``--out-dir data_msf`` 表示当前工作目录下的 ``data_msf``）。

用法示例::

    # 整目录（输出到默认 data/data_msf）
    python3 msf/export_mscanfit.py --input-dir data/data_70w

    # 指定输出目录
    python3 msf/export_mscanfit.py --input-dir data/data_70w --out-dir data/data_msf

    python3 msf/export_mscanfit.py --mat data/data_70w/500_70w.mat --out-dir /path/to/custom

依赖：h5py（``pip install h5py``）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "data_msf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 v7.3 .mat 按 label_num 抽样导出 MScanFit 双列 txt。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--input-dir",
        type=Path,
        help="包含若干 .mat 的目录；会处理目录内所有原始 .mat（排除 *_train/_val/_test）",
    )
    g.add_argument(
        "--mat",
        type=Path,
        nargs="+",
        metavar="PATH",
        help="一个或多个 .mat 文件路径",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="导出 txt 的根目录；默认 data/data_msf（也可用 data_msf 等任意路径）",
    )
    parser.add_argument(
        "--labels",
        type=float,
        nargs="+",
        default=[50.0, 100.0, 150.0],
        help="要导出的 label_num 取值",
    )
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=20,
        help="每个 label 最多导出多少条曲线",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="文件名前缀；默认使用各 .mat 文件名（不含扩展名）",
    )
    parser.add_argument(
        "--fmt",
        default="%.15g",
        help="两列浮点格式，传给 numpy.savetxt（每列一个 fmt）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将处理哪些文件与样本数，不写文件",
    )
    return parser.parse_args()


def _collect_mat_paths(input_dir: Path | None, mat_paths: list[Path] | None) -> list[Path]:
    if mat_paths:
        out = []
        for p in mat_paths:
            p = p.expanduser().resolve()
            if not p.is_file():
                raise FileNotFoundError(f"找不到文件: {p}")
            if p.suffix.lower() != ".mat":
                raise ValueError(f"不是 .mat: {p}")
            out.append(p)
        return out

    assert input_dir is not None
    d = input_dir.expanduser().resolve()
    if not d.is_dir():
        raise NotADirectoryError(f"不是目录: {d}")
    mats = sorted(d.glob("*.mat"))
    mats = [
        p
        for p in mats
        if not (
            p.stem.endswith("_train")
            or p.stem.endswith("_val")
            or p.stem.endswith("_test")
        )
    ]
    if not mats:
        raise FileNotFoundError(f"目录下没有可处理的 .mat: {d}")
    return mats


def _labels_1d(f: h5py.File) -> np.ndarray:
    if "label_num" not in f:
        raise KeyError("mat 中缺少数据集 'label_num'")
    y = np.asarray(f["label_num"][:], dtype=np.float64).squeeze()
    if y.ndim != 1:
        y = y.reshape(-1)
    return y


def _indices_for_label(y: np.ndarray, target: float, max_count: int) -> np.ndarray:
    """与 mat 中 label 匹配的下标（支持近似整数标签）。"""
    if max_count <= 0:
        return np.array([], dtype=np.int64)
    # 与整型标签常用写法一致：四舍五入后比较
    yi = np.rint(y)
    mask = yi == float(np.rint(target))
    idx = np.flatnonzero(mask)
    return idx[:max_count]


def export_one_mat(
    mat_path: Path,
    out_root: Path,
    labels: list[float],
    samples_per_label: int,
    stem: str,
    col_fmt: str,
    dry_run: bool,
) -> int:
    """导出单个 mat；文件名为 ``{stem}_{label}_{j}_{k}.txt``（j=真实列下标 0 起，k=该 label 下导出序号 1 起）。"""
    if not dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    written = 0
    with h5py.File(mat_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"{mat_path}: 缺少数据集 'data'")
        dset = f["data"]
        if dset.ndim != 3 or dset.shape[0] != 2:
            raise ValueError(
                f"{mat_path}: data 形状应为 (2, L, N)，当前为 {tuple(dset.shape)}"
            )
        n_samples = int(dset.shape[2])
        y = _labels_1d(f)
        if y.shape[0] != n_samples:
            raise ValueError(
                f"{mat_path}: label_num 长度 {y.shape[0]} 与 data 样本维 {n_samples} 不一致"
            )

        for lab in labels:
            idxs = _indices_for_label(y, lab, samples_per_label)
            if idxs.size == 0:
                print(f"  [skip] {mat_path.name}: label={lab:g} 无匹配样本", file=sys.stderr)
                continue
            if idxs.size < samples_per_label:
                print(
                    f"  [warn] {mat_path.name}: label={lab:g} 仅 {idxs.size} 条（少于 --samples-per-label）",
                    file=sys.stderr,
                )

            for k, j in enumerate(idxs, start=1):
                # 只读一列，避免整文件进内存
                xy = np.asarray(dset[:, :, j], dtype=np.float64)
                x = xy[0, :].ravel()
                ycol = xy[1, :].ravel()
                out_path = out_root / f"{stem}_{lab:g}_{int(j)}_{k}.txt"
                if dry_run:
                    print(f"  would write {out_path} (n={x.size})")
                else:
                    _write_mscanfit_txt(out_path, x, ycol, col_fmt)
                written += 1

    return written


def _write_mscanfit_txt(path: Path, x: np.ndarray, y: np.ndarray, col_fmt: str) -> None:
    """两列、TAB、CRLF。"""
    if x.shape != y.shape:
        raise ValueError(f"x/y 长度不一致: {x.shape} vs {y.shape}")
    data = np.column_stack([x, y])
    np.savetxt(
        path,
        data,
        fmt=(col_fmt, col_fmt),
        delimiter="\t",
        newline="\r\n",
    )


def main() -> None:
    args = parse_args()
    out_root = args.out_dir.expanduser().resolve()
    mats = _collect_mat_paths(args.input_dir, args.mat)

    total = 0
    print(f"输出根目录: {out_root}")
    print(f"待处理 .mat 数量: {len(mats)}")
    for mat_path in mats:
        print(f"处理: {mat_path}")
        if args.prefix is not None:
            if len(mats) > 1:
                stem = f"{args.prefix}_{mat_path.stem}"
            else:
                stem = args.prefix
        else:
            stem = mat_path.stem

        n = export_one_mat(
            mat_path,
            out_root,
            list(args.labels),
            args.samples_per_label,
            stem,
            args.fmt,
            args.dry_run,
        )
        total += n
        print(f"  -> 写出 {n} 个 txt")

    print(f"完成。共 {'将写入' if args.dry_run else '写入'} {total} 个文件。")


if __name__ == "__main__":
    main()
