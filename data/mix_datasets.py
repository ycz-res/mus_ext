"""
多份 HDF5 .mat（与 tools.load_data 一致：data / label_num，可选其它键）合并为一份。

流程（仅此一种）：
  1. 对每个输入文件：**按四舍五入后的整数 MU 分层抽样**，每个 MU 取
     ``min(该 MU 条数, max(1, round(该 MU 条数 * --fraction)))``；
  2. **按 --inputs 顺序**把各文件样本沿样本维拼接，不同序列长 L 写入 ``seq_len``，``data`` 为 ``(2, max_L, N)``；
  3. 对合并结果 **shuffle**（``--seed``），再写出 ``--output``。

例如各档 ``*_70w.mat`` 每 MU 5000 条、``--fraction 0.2``、6 个文件：每文件每 MU 1000 条，
合并后每 MU 约 6000 条，总样本约 ``156 * 6000``。

依赖：h5py、numpy。
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
    return y.astype(np.float32).reshape(-1)


def _pick_concat_axis(arrays: list[np.ndarray]) -> int:
    ref_shape = arrays[0].shape
    if len(arrays) == 1:
        return 0
    diff_axes: set[int] = set()
    for arr in arrays[1:]:
        if arr.ndim != len(ref_shape):
            raise ValueError("All arrays for a key must have the same rank.")
        for i, (a, b) in enumerate(zip(ref_shape, arr.shape)):
            if a != b:
                diff_axes.add(i)
    if len(diff_axes) == 0:
        return 0
    if len(diff_axes) > 1:
        raise ValueError(
            f"Cannot infer concat axis: multiple dimensions differ ({sorted(diff_axes)})."
        )
    return next(iter(diff_axes))


def _infer_sample_axis_for_n(arr: np.ndarray, n_samples: int) -> int | None:
    candidates = [i for i, s in enumerate(arr.shape) if int(s) == n_samples]
    if len(candidates) == 1:
        return candidates[0]
    return None


def subsample_mat_dict_per_mu(
    data: dict[str, np.ndarray],
    fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """单文件：按整数 MU 分层抽样；每 MU 取 min(条数, max(1, round(条数 * fraction)))。"""
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction 须在 (0, 1]，当前为 {fraction}")

    arr_data = np.asarray(data["data"], dtype=np.float32)
    if arr_data.ndim != 3 or arr_data.shape[0] != 2:
        raise ValueError(f"data 期望 (2, L, N)，当前 {arr_data.shape}")
    n = int(arr_data.shape[2])
    y = _label_to_1d(data["label_num"])
    if y.size != n:
        raise ValueError(f"label 样本数 {y.size} 与 data 的 N={n} 不一致")

    yi = np.rint(y.astype(np.float64)).astype(np.int64)
    rng = np.random.default_rng(seed)
    picked_parts: list[np.ndarray] = []
    for mu in np.unique(yi):
        idx_all = np.flatnonzero(yi == mu)
        c = int(idx_all.size)
        n_take = max(1, int(round(c * fraction)))
        n_take = min(n_take, c)
        perm = rng.permutation(c)[:n_take]
        picked_parts.append(idx_all[perm])
    idx = np.concatenate(picked_parts)
    rng.shuffle(idx)

    out: dict[str, np.ndarray] = {}
    for key, arr in data.items():
        arr = np.asarray(arr)
        axis = _infer_sample_axis_for_n(arr, n)
        if axis is None:
            out[key] = arr
        else:
            out[key] = np.take(arr, idx, axis=axis)
    return out


def merge_concat_samples(
    loaded: list[dict[str, np.ndarray]],
    max_seq_len: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """按文件顺序沿样本维拼接；各文件 L 可不同，写入 seq_len，data 为 (2, max_L, N)。"""
    keys = set(loaded[0].keys())
    for d in loaded[1:]:
        keys &= set(d.keys())
    if "data" not in keys or "label_num" not in keys:
        raise ValueError("需要 'data' 与 'label_num'。")

    data_arrays = [np.asarray(d["data"], dtype=np.float32) for d in loaded]
    if not all(a.ndim == 3 and a.shape[0] == 2 for a in data_arrays):
        raise ValueError("data 期望形状 (2, L, N)。")

    per_file_L: list[int] = []
    for a in data_arrays:
        L = int(a.shape[1])
        if max_seq_len is not None:
            L = min(L, int(max_seq_len))
        per_file_L.append(L)

    max_L = max(per_file_L)
    n_list = [a.shape[2] for a in data_arrays]
    N_total = sum(n_list)

    labels_1d: list[np.ndarray] = []
    for i, d in enumerate(loaded):
        y = _label_to_1d(d["label_num"])
        n = data_arrays[i].shape[2]
        if y.size != n:
            raise ValueError(
                f"第 {i} 个文件: label 样本数 {y.size} 与 data 的 N={n} 不一致。"
            )
        labels_1d.append(y)

    seq_blocks: list[int] = []
    merged_data = np.zeros((2, max_L, N_total), dtype=np.float32)
    col = 0
    for a, L_i, n_f in zip(data_arrays, per_file_L, n_list):
        merged_data[:, :L_i, col : col + n_f] = a[:, :L_i, :]
        seq_blocks.extend([L_i] * n_f)
        col += n_f

    merged_seq = np.asarray(seq_blocks, dtype=np.float32).reshape(1, -1)
    merged_label = np.concatenate(
        [y.reshape(1, -1) for y in labels_1d],
        axis=1,
    ).astype(np.float32)

    merged: dict[str, np.ndarray] = {
        "data": merged_data,
        "label_num": merged_label,
        "seq_len": merged_seq,
    }
    axis_by_key: dict[str, int] = {
        "data": 2,
        "label_num": 1,
        "seq_len": 1,
    }

    for key in sorted(keys):
        if key in merged:
            continue
        arrs = [np.asarray(d[key]) for d in loaded]
        if all(
            x.ndim == 3 and x.shape[0] == 2 and x.shape[1] == data_arrays[i].shape[1]
            for i, x in enumerate(arrs)
        ):
            out = np.zeros((2, max_L, N_total), dtype=np.float32)
            col = 0
            for a, L_i, n_f in zip(arrs, per_file_L, n_list):
                out[:, :L_i, col : col + n_f] = a[:, :L_i, :]
                col += n_f
            merged[key] = out
            axis_by_key[key] = 2
        else:
            axis = _pick_concat_axis(arrs)
            merged[key] = np.concatenate(arrs, axis=axis)
            axis_by_key[key] = axis

    _print_shapes("concat_samples", merged, axis_by_key)
    return merged, axis_by_key


def _print_shapes(tag: str, merged: dict, axis_by_key: dict) -> None:
    for key in sorted(merged.keys()):
        print(
            f"[{key}] {tag} -> {merged[key].shape} "
            f"(sample axis={axis_by_key.get(key, '?')})"
        )


def shuffle_merged(
    merged: dict[str, np.ndarray],
    axis_by_key: dict[str, int],
    seed: int | None,
) -> dict[str, np.ndarray]:
    """沿样本维随机打乱；各键样本维长度一致的一并重排。"""
    ref_key = "label_num" if "label_num" in merged else next(iter(merged.keys()))
    ref_axis = axis_by_key[ref_key]
    n_samples = merged[ref_key].shape[ref_axis]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    print(f"\nShuffle: seed={seed}, n_samples={n_samples}")

    out: dict[str, np.ndarray] = {}
    for key, value in merged.items():
        axis = axis_by_key[key]
        if value.shape[axis] == n_samples:
            out[key] = np.take(value, indices, axis=axis)
        else:
            out[key] = value
    return out


def save_merged(merged: dict[str, np.ndarray], output_path: Path) -> None:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        for key, value in merged.items():
            f.create_dataset(key, data=value)

    print(f"\nMerged file saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="多 .mat：每文件按整数 MU 取比例 → 拼接混合 → shuffle 写出（seq_len 变长）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例（六档 70w，每文件每 MU 取 1/5，再混合；约 156*6000 条）:
  python3 data/mix_datasets.py \\
    --inputs data/data_70w/500_70w.mat data/data_70w/600_70w.mat data/data_70w/700_70w.mat \\
            data/data_70w/800_70w.mat data/data_70w/900_70w.mat data/data_70w/1000_70w.mat \\
    --output data/mix_70w_f0p2.mat --fraction 0.2 --max-seq-len 1000 --seed 42
""",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="按顺序列出输入 .mat（至少 2 个）",
    )
    parser.add_argument("--output", required=True, help="输出 .mat")
    parser.add_argument(
        "--fraction",
        type=float,
        required=True,
        metavar="F",
        help="每文件内每个整数 MU 保留比例：每 MU 取 min(条数, max(1, round(条数*F)))",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="抽样与 shuffle 共用种子（每文件抽样 seed = seed*1000003 + 文件序号）",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        metavar="L",
        help="合并前各文件 data 在时间维截断到此长度",
    )
    args = parser.parse_args()

    loaded: list[dict[str, np.ndarray]] = []
    for fp in [Path(p).resolve() for p in args.inputs]:
        if not fp.exists():
            raise FileNotFoundError(f"Input file not found: {fp}")
        with h5py.File(fp, "r") as f:
            loaded.append({k: np.array(f[k]) for k in f.keys()})

    if len(loaded) < 2:
        raise ValueError("至少需要 2 个输入文件。")

    loaded = [
        subsample_mat_dict_per_mu(d, args.fraction, args.seed * 1_000_003 + i)
        for i, d in enumerate(loaded)
    ]
    print(
        f"\nPer-file per-MU subsample: fraction={args.fraction}, seed_base={args.seed}, "
        f"n_files={len(loaded)}"
    )

    merged, axis_by_key = merge_concat_samples(loaded, max_seq_len=args.max_seq_len)
    merged = shuffle_merged(merged, axis_by_key, args.seed)
    save_merged(merged, args.output)


if __name__ == "__main__":
    main()
