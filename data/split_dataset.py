"""
将单个 HDF5 .mat 数据集按样本维切分为 train/val/test。

默认比例 8:1:1，输出：
  xxx_train.mat
  xxx_val.mat
  xxx_test.mat

兼容本仓库常见字段：
  - data: (2, L, N)    -> 样本轴通常是 2
  - label_num: (1, N)  -> 样本轴通常是 1
  - seq_len: (1, N)    -> 样本轴通常是 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _label_to_1d(label_arr: np.ndarray) -> np.ndarray:
    y = np.asarray(label_arr).squeeze()
    if y.ndim == 2:
        y = y[0] if y.shape[0] < y.shape[1] else y[:, 0]
    return y.reshape(-1)


def _infer_sample_count(data: dict[str, np.ndarray]) -> int:
    if "label_num" in data:
        n = int(_label_to_1d(data["label_num"]).shape[0])
        if n <= 0:
            raise ValueError("label_num 为空，无法推断样本数。")
        return n

    # 回退策略：使用最大维度作为样本数（仅在无 label_num 时使用）
    max_dim = 0
    for v in data.values():
        if np.asarray(v).ndim > 0:
            max_dim = max(max_dim, int(max(np.asarray(v).shape)))
    if max_dim <= 0:
        raise ValueError("无法从输入 .mat 推断样本数。")
    return max_dim


def _infer_sample_axis(arr: np.ndarray, n_samples: int) -> int | None:
    candidates = [i for i, s in enumerate(arr.shape) if int(s) == n_samples]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _split_indices(n: int, train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("划分比例之和必须大于 0。")
    train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    # 保证总数精确为 n
    if n_train + n_val > n:
        n_val = n - n_train
    n_test = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val : n_train + n_val + n_test]
    return train_idx, val_idx, test_idx


def _slice_by_indices(arr: np.ndarray, axis: int, idx: np.ndarray) -> np.ndarray:
    return np.take(arr, idx, axis=axis)


def _build_split(data: dict[str, np.ndarray], n_samples: int, idx: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, arr in data.items():
        arr = np.asarray(arr)
        axis = _infer_sample_axis(arr, n_samples)
        if axis is None:
            # 对无法识别样本轴的字段，原样拷贝
            out[key] = arr
        else:
            out[key] = _slice_by_indices(arr, axis, idx)
    return out


def _save_mat(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in data.items():
            f.create_dataset(k, data=v)


def split_mat_file(
    input_path: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    output_dir: Path | None = None,
) -> tuple[Path, Path, Path]:
    input_path = input_path.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    if input_path.suffix.lower() != ".mat":
        raise ValueError("输入文件必须是 .mat")

    with h5py.File(input_path, "r") as f:
        loaded = {k: np.array(f[k]) for k in f.keys()}

    n_samples = _infer_sample_count(loaded)
    train_idx, val_idx, test_idx = _split_indices(
        n_samples, train_ratio, val_ratio, test_ratio, seed
    )

    train_data = _build_split(loaded, n_samples, train_idx)
    val_data = _build_split(loaded, n_samples, val_idx)
    test_data = _build_split(loaded, n_samples, test_idx)

    stem = input_path.stem
    out_dir = (output_dir if output_dir is not None else input_path.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f"{stem}_train.mat"
    val_path = out_dir / f"{stem}_val.mat"
    test_path = out_dir / f"{stem}_test.mat"

    _save_mat(train_path, train_data)
    _save_mat(val_path, val_data)
    _save_mat(test_path, test_data)

    print(f"input: {input_path}")
    print(f"samples: total={n_samples}, train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"saved: {train_path}")
    print(f"saved: {val_path}")
    print(f"saved: {test_path}")

    return train_path, val_path, test_path


def split_all_mat_files_in_dir(
    data_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    output_dir: Path | None = None,
) -> None:
    data_dir = data_dir.resolve()
    if not data_dir.exists() or not data_dir.is_dir():
        raise NotADirectoryError(f"目录不存在或不是目录: {data_dir}")

    mat_files = sorted(data_dir.glob("*.mat"))
    # 避免把已经切分过的文件再次切分
    mat_files = [
        p
        for p in mat_files
        if not (p.stem.endswith("_train") or p.stem.endswith("_val") or p.stem.endswith("_test"))
    ]
    if not mat_files:
        print(f"目录 {data_dir} 下没有可切分的 .mat 文件。")
        return

    print(f"发现 {len(mat_files)} 个 .mat 文件，开始按 8:1:1 切分...")
    for p in mat_files:
        print("-" * 60)
        split_mat_file(
            p,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            output_dir=output_dir,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="将 .mat 按 8:1:1 切分为 train/val/test。")
    parser.add_argument("input", nargs="?", default=None, help="输入 .mat 文件路径，例如 data/xxx.mat")
    parser.add_argument("--all-in-dir", default=None, help="批量切分目录下所有 .mat，例如 data")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例，默认 0.8")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例，默认 0.1")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例，默认 0.1")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="train/val/test .mat 输出目录；默认与输入文件同目录",
    )
    args = parser.parse_args()

    if args.all_in_dir is not None:
        split_all_mat_files_in_dir(
            Path(args.all_in_dir),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        return

    if args.input is None:
        parser.error("请提供 input 文件，或使用 --all-in-dir 批量处理目录。")

    split_mat_file(
        Path(args.input),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
