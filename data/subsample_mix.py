#!/usr/bin/env python3
"""
从多个「固定序列长度」的 test .mat 中，各随机抽取相同比例（默认 1/5）的样本，
再按样本维合并为单个混合测试集（与 mix_datasets 一致：写入 seq_len，供 tools.load_data 变长读取）。

示例（混合 .mat 放在 data/xl_mix_sets/ 根目录，须指定 --output 文件名）:
  python3 data/subsample_mix.py \\
    --out-dir data/xl_mix_sets \\
    --output mix_no500_test.mat \\
    --inputs data/split/600_10w_test.mat data/split/700_10w_test.mat \\
             data/split/800_10w_test.mat data/split/900_10w_test.mat \\
             data/split/1000_10w_test.mat \\
    --fraction 0.2 --seed 42 --shuffle-seed 42
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np

# 与 main 运行时 cwd 无关：同目录下 mix_datasets
_DATA_DIR = Path(__file__).resolve().parent
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))

import mix_datasets as mix  # noqa: E402


def _label_n_samples(label_arr: np.ndarray) -> int:
    y = np.asarray(label_arr).squeeze()
    if y.ndim == 2:
        y = y[0] if y.shape[0] < y.shape[1] else y[:, 0]
    return int(y.reshape(-1).shape[0])


def _sample_indices(n: int, fraction: float, seed: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("样本数为 0")
    n_take = max(1, int(round(n * fraction)))
    n_take = min(n_take, n)
    rng = np.random.default_rng(seed)
    return rng.permutation(n)[:n_take]


def _slice_mat_dict(data: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    """按样本维切片；约定 data (2,L,N)，label_num (1,N)，可选 seq_len (1,N)。"""
    n = _label_n_samples(data["label_num"])
    if idx.max() >= n or idx.min() < 0:
        raise IndexError("抽样索引越界")

    out: dict[str, np.ndarray] = {}
    for key, arr in data.items():
        arr = np.asarray(arr)
        if key == "data":
            if arr.ndim != 3 or arr.shape[0] != 2:
                raise ValueError(f"data 期望 (2, L, N)，当前 {arr.shape}")
            if arr.shape[2] != n:
                raise ValueError(f"data N={arr.shape[2]} 与 label 样本数 {n} 不一致")
            out[key] = arr[:, :, idx]
        elif key == "label_num":
            if arr.ndim == 2:
                out[key] = arr[:, idx]
            else:
                out[key] = arr[idx]
        elif key == "seq_len":
            if arr.ndim == 2 and arr.shape[1] == n:
                out[key] = arr[:, idx]
            elif arr.ndim == 1 and arr.shape[0] == n:
                out[key] = arr[idx]
            else:
                raise ValueError(f"无法按样本切 seq_len: {arr.shape}, n={n}")
        else:
            # 其它字段：若能识别样本轴则切，否则原样拷贝
            axes = [i for i, s in enumerate(arr.shape) if int(s) == n]
            if len(axes) == 1:
                out[key] = np.take(arr, idx, axis=axes[0])
            else:
                out[key] = arr
    return out


def load_mat(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        return {k: np.array(f[k]) for k in f.keys()}


def _default_mat_filename(out_dir: Path) -> str:
    """目录名 xl_mix_no500 -> mix_no500_test.mat（仍支持临时子目录名）。"""
    m = re.match(r"^xl_mix_(.+)$", out_dir.name)
    if m:
        return f"mix_{m.group(1)}_test.mat"
    return "mix_test.mat"


def main() -> None:
    parser = argparse.ArgumentParser(description="多长度 test 各抽一定比例后合并为混合测试 .mat")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="按顺序列出输入 .mat（如 600/700/800/900/1000 的 *_test.mat）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="输出目录（将新建）；根目录为 xl_mix_sets 时必须配合 --output；子目录 xl_mix_no* 可自动推断文件名",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="NAME.mat",
        help="输出 .mat 文件名（可不含 .mat）；不设则按 --out-dir 最后一级目录名推断",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.2,
        help="每个文件抽取的样本比例，默认 0.2（五分之一）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="各文件抽样的随机种子基数；第 i 个文件实际 seed = seed + i",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="合并后打乱顺序的种子，默认与 --seed 相同",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="传入 merge 前的截断上限；一般 6h–1k（600–1000）可不设",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="合并后不打乱（顺序为文件1子集、文件2子集……）",
    )
    args = parser.parse_args()

    if len(args.inputs) < 2:
        raise SystemExit("至少需要 2 个输入文件才能合并")

    shuffle_seed = args.seed if args.shuffle_seed is None else args.shuffle_seed
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_dir.name == "xl_mix_sets" and not args.output:
        raise SystemExit(
            "--out-dir 为 xl_mix_sets 时必须指定 --output（如 mix_no600_test.mat），避免覆盖根目录下其它 .mat"
        )

    loaded: list[dict[str, np.ndarray]] = []
    manifest_lines: list[str] = []

    for i, p in enumerate(args.inputs):
        path = Path(p).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        raw = load_mat(path)
        n_full = _label_n_samples(raw["label_num"])
        idx = _sample_indices(n_full, args.fraction, args.seed + i)
        sub = _slice_mat_dict(raw, idx)
        loaded.append(sub)
        manifest_lines.append(
            f"{path.name}: n_full={n_full}, n_sampled={len(idx)}, seed={args.seed + i}"
        )
        print(manifest_lines[-1])

    merged, axis_by_key = mix.merge_concat_samples(loaded, max_seq_len=args.max_seq_len)
    if not args.no_shuffle:
        merged = mix.shuffle_merged(merged, axis_by_key, shuffle_seed)

    out_name = args.output or _default_mat_filename(out_dir)
    if not out_name.endswith(".mat"):
        out_name = f"{out_name}.mat"
    out_mat = out_dir / out_name
    mix.save_merged(merged, out_mat)

    n_total = merged["label_num"].shape[1]
    manifest_lines.append(f"merged_total_samples={n_total}")
    manifest_lines.append(f"fraction={args.fraction}")
    manifest_lines.append(f"shuffle={'none' if args.no_shuffle else f'seed={shuffle_seed}'}")
    manifest_lines.append(f"output={out_mat.name}")

    manifest_path = out_dir / "manifest.txt"
    block = "\n".join([f"========== {out_mat.stem} ==========", *manifest_lines, ""])
    if out_dir.name == "xl_mix_sets":
        with manifest_path.open("a", encoding="utf-8") as mf:
            mf.write(block)
    else:
        manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
