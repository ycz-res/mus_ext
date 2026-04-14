"""
合并多个 HDF5 .mat（与 tools.load_data 布局一致）。

「混合」只做一件事：**按 --inputs 顺序做样本组装**——先第 1 个文件里的全部样本，
再第 2 个文件……例如 500 长的文件里是样本 1,2,3，1000 长的文件里是 4,5,6，合并后样本顺序为
1,2,3,4,5,6；其中 **1–3 仍是 500 个时间步**，**4–6 仍是 1000 个时间步**（不是把两条波形沿时间拼成一条）。

实现上 `data` 存为 `(2, max_L, N)`，`max_L` 为各文件有效长度（经 `--max-seq-len` 截断后）的最大值；
较短样本只占列的前 `seq_len[i]` 个时间步，列尾未用位置为 0（仅存储占位，语义见 `seq_len`）。
另存 **`seq_len`**，形状 `(1, N)`，每个样本的真实长度，供 `tools.load_data` 使用。

合并完成后可对**样本顺序**做打乱（`--order shuffle`，CLI 默认）；`data`、`label_num`、`seq_len`
等同索引一起重排。

约定:
  - data: (2, max_L, N)，label_num: (1, N)，seq_len: (1, N) 每个样本有效长度 L
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


def merge_concat_samples(
    loaded: list[dict[str, np.ndarray]],
    max_seq_len: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """
    单纯样本组装：按文件顺序沿样本维拼接。各文件 L 可不同；按列写入 (2, max_L, N)，
    不将短序列用零「拉长」到长序列长度（仅矩阵列尾未用位置为 0），并写入 seq_len。
    """
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


def _concat_label_num(arrays: list[np.ndarray]) -> tuple[np.ndarray, int]:
    pieces: list[np.ndarray] = []
    for a in arrays:
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        elif a.ndim == 2:
            a = a[0:1, :] if a.shape[0] >= 1 else a
        else:
            raise ValueError(f"label_num 无法解析: {a.shape}")
        pieces.append(a)
    return np.concatenate(pieces, axis=1), 1


def merge_mat_files_auto(
    loaded: list[dict[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    common_keys = set(loaded[0].keys())
    for d in loaded[1:]:
        common_keys &= set(d.keys())
    if not common_keys:
        raise ValueError("No common keys found across input files.")

    merged: dict[str, np.ndarray] = {}
    axis_by_key: dict[str, int] = {}

    for key in sorted(common_keys):
        arrays = [d[key] for d in loaded]
        if key == "label_num":
            merged[key], axis_by_key[key] = _concat_label_num(arrays)
            print(
                f"[{key}] concat sample axis=1, "
                f"shapes={[np.asarray(a).shape for a in arrays]} -> {merged[key].shape}"
            )
            continue
        axis = _pick_concat_axis(arrays)
        axis_by_key[key] = axis
        merged[key] = np.concatenate(arrays, axis=axis)
        print(
            f"[{key}] concat axis={axis}, "
            f"shapes={[np.asarray(a).shape for a in arrays]} -> {merged[key].shape}"
        )

    return merged, axis_by_key


def merge_mat_files(
    input_paths: list[str | Path],
    output_path: str | Path,
    strategy: str = "mix",
    max_seq_len: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    files = [Path(p).resolve() for p in input_paths]
    if len(files) < 2:
        raise ValueError("至少需要 2 个输入文件。")
    for fp in files:
        if not fp.exists():
            raise FileNotFoundError(f"Input file not found: {fp}")

    loaded: list[dict[str, np.ndarray]] = []
    for fp in files:
        with h5py.File(fp, "r") as f:
            loaded.append({k: np.array(f[k]) for k in f.keys()})

    if strategy in ("mix", "concat_samples", "stack_samples", "train", "pad_samples"):
        return merge_concat_samples(loaded, max_seq_len=max_seq_len)
    if strategy == "auto":
        return merge_mat_files_auto(loaded)
    raise ValueError(f"Unknown strategy: {strategy}")


def _print_shapes(tag: str, merged: dict, axis_by_key: dict) -> None:
    for key in sorted(merged.keys()):
        print(
            f"[{key}] {tag} -> {merged[key].shape} "
            f"(sample axis={axis_by_key.get(key, '?')})"
        )


def reorder_merged(
    merged: dict[str, np.ndarray],
    axis_by_key: dict[str, int],
    order_mode: str,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    if order_mode == "none":
        return merged

    ref_key = "label_num" if "label_num" in merged else next(iter(merged.keys()))
    ref_axis = axis_by_key[ref_key]
    n_samples = merged[ref_key].shape[ref_axis]

    if order_mode == "shuffle":
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_samples)
        print(f"\nReorder: shuffle, seed={seed}, n_samples={n_samples}")
    elif order_mode == "sort":
        if "label_num" not in merged:
            print("\nReorder: sort skipped (no label_num key).")
            return merged
        lab = merged["label_num"]
        ax = axis_by_key["label_num"]
        labels = np.take(lab, np.arange(lab.shape[ax]), axis=ax).reshape(-1)
        indices = np.argsort(labels, kind="stable")
        print("\nReorder: sort by label_num (ascending).")
    else:
        raise ValueError(f"Unknown order mode: {order_mode}")

    reordered: dict[str, np.ndarray] = {}
    for key, value in merged.items():
        axis = axis_by_key[key]
        if value.shape[axis] == n_samples:
            reordered[key] = np.take(value, indices, axis=axis)
        else:
            reordered[key] = value
    return reordered


def save_merged(merged: dict[str, np.ndarray], output_path: Path) -> None:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        for key, value in merged.items():
            f.create_dataset(key, data=value)

    print(f"\nMerged file saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按文件顺序拼接样本；各文件 L 可不同，写入 seq_len 表示每条样本真实长度。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例（先 500 长度集全部样本，再 1000 长度集全部样本；约 40w 条）:
  python3 data/mix_datasets.py --inputs data/train_500_20w.mat data/train_1000_20w.mat \\
    --output data/train_mix_40w.mat --max-seq-len 1000 --order shuffle --seed 42

验证集（约 3.2w 条）:
  python3 data/mix_datasets.py --inputs data/dev_500_1.6w.mat data/dev_1000_1.6w.mat \\
    --output data/dev_mix_32k.mat --max-seq-len 1000 --order shuffle --seed 42
""",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="按顺序列出 .mat；样本顺序为先第 1 个文件，再第 2 个……",
    )
    parser.add_argument("--output", required=True, help="输出 .mat")
    parser.add_argument(
        "--order",
        choices=["shuffle", "sort", "none"],
        default="shuffle",
        help="合并后是否打乱样本（训练常用 shuffle）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="shuffle 随机种子。",
    )
    parser.add_argument(
        "--strategy",
        choices=["mix", "concat_samples", "auto"],
        default="mix",
        help="mix: 样本维拼接（默认）；auto: 按 key 自动推断轴。",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        metavar="L",
        help="各文件截断到此长度后再合并；输出 data 宽为各文件有效 L 的最大值。",
    )
    args = parser.parse_args()

    merged, axis_by_key = merge_mat_files(
        args.inputs,
        args.output,
        strategy=args.strategy,
        max_seq_len=args.max_seq_len,
    )
    merged = reorder_merged(merged, axis_by_key, args.order, args.seed)
    save_merged(merged, args.output)


if __name__ == "__main__":
    main()
