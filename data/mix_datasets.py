import argparse
from pathlib import Path

import h5py
import numpy as np


def _pick_concat_axis(arrays):
    """Infer concat axis by finding the only dimension that differs."""
    ref_shape = arrays[0].shape
    if len(arrays) == 1:
        return 0

    diff_axes = set()
    for arr in arrays[1:]:
        if arr.ndim != len(ref_shape):
            raise ValueError("All arrays for a key must have the same rank.")
        for i, (a, b) in enumerate(zip(ref_shape, arr.shape)):
            if a != b:
                diff_axes.add(i)

    if len(diff_axes) == 0:
        # Same shape everywhere; keep behavior simple.
        return 0
    if len(diff_axes) > 1:
        raise ValueError(
            f"Cannot infer concat axis: multiple dimensions differ ({sorted(diff_axes)})."
        )
    return next(iter(diff_axes))


def merge_mat_files(input_paths, output_path):
    files = [Path(p).resolve() for p in input_paths]
    if len(files) < 2:
        raise ValueError("Please provide at least 2 input files.")
    for fp in files:
        if not fp.exists():
            raise FileNotFoundError(f"Input file not found: {fp}")

    loaded = []
    for fp in files:
        with h5py.File(fp, "r") as f:
            loaded.append({k: np.array(f[k]) for k in f.keys()})

    common_keys = set(loaded[0].keys())
    for d in loaded[1:]:
        common_keys &= set(d.keys())
    if not common_keys:
        raise ValueError("No common keys found across input files.")

    merged = {}
    axis_by_key = {}
    for key in sorted(common_keys):
        arrays = [d[key] for d in loaded]
        axis = _pick_concat_axis(arrays)
        axis_by_key[key] = axis
        merged[key] = np.concatenate(arrays, axis=axis)
        print(
            f"[{key}] concat axis={axis}, "
            f"shapes={[a.shape for a in arrays]} -> {merged[key].shape}"
        )

    return merged, axis_by_key


def reorder_merged(merged, axis_by_key, order_mode, seed=None):
    if order_mode == "none":
        return merged

    ref_key = "label_num" if "label_num" in merged else next(iter(merged.keys()))
    ref_axis = axis_by_key[ref_key]
    n_samples = merged[ref_key].shape[ref_axis]

    if order_mode == "shuffle":
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_samples)
        print(f"\nReorder: shuffle, seed={seed}")
    elif order_mode == "sort":
        if "label_num" not in merged:
            print("\nReorder: sort skipped (no label_num key).")
            return merged
        labels = np.take(
            merged["label_num"],
            np.arange(merged["label_num"].shape[axis_by_key["label_num"]]),
            axis=axis_by_key["label_num"],
        ).reshape(-1)
        indices = np.argsort(labels, kind="stable")
        print("\nReorder: sort by label_num (ascending).")
    else:
        raise ValueError(f"Unknown order mode: {order_mode}")

    reordered = {}
    for key, value in merged.items():
        axis = axis_by_key[key]
        if value.shape[axis] == n_samples:
            reordered[key] = np.take(value, indices, axis=axis)
        else:
            reordered[key] = value
    return reordered


def save_merged(merged, output_path):
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        for key, value in merged.items():
            f.create_dataset(key, data=value)

    print(f"\nMerged file saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple .mat (HDF5) dataset files by shared keys."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input .mat paths, e.g. --inputs a.mat b.mat c.mat",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .mat path, e.g. --output data/train_mix.mat",
    )
    parser.add_argument(
        "--order",
        choices=["shuffle", "sort", "none"],
        default="shuffle",
        help="How to reorder merged samples (default: shuffle).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --order shuffle (default: 42).",
    )
    args = parser.parse_args()

    merged, axis_by_key = merge_mat_files(args.inputs, args.output)
    merged = reorder_merged(merged, axis_by_key, args.order, args.seed)
    save_merged(merged, args.output)


if __name__ == "__main__":
    main()
