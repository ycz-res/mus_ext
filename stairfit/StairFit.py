#!/usr/bin/env python3
"""
Batch runner for the original MATLAB StairFit.m.

Default behavior:
  - input:  data_msf/
  - output: results/res_sf_70w/stairfit_results.xlsx
  - fitting: calls MATLAB StairFit.m, not a Python reimplementation

Resume (断电续跑):
  - 每个 txt 的结果写在 output-dir/_matlab_runs/run_stairfit_<stem>.json
  - 再次运行同一命令时，若 json 已存在且 status=ok，则跳过 MATLAB，直接载入
  - 每完成（或跳过）一个文件会更新 stairfit_results.xlsx，避免断电丢全部结果
  - 强制重算全部：加 --no-resume

Examples:
  python3 StairFit.py
  python3 StairFit.py data_msf
  python3 StairFit.py data_msf --file 500_70w_50_225000_1.txt
  python3 StairFit.py data_msf --pattern "500_70w_50_*.txt"
  python3 StairFit.py data_msf --stairfit-dir ./StairFit
  python3 StairFit.py data_msf --matlab-cmd /usr/local/MATLAB/R2024a/bin/matlab
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


def resolve_input_files(input_path: Path, file_name: str | None, pattern: str | None) -> list[Path]:
    path = input_path.expanduser().resolve()

    if path.is_file():
        if file_name or pattern:
            raise ValueError("--file and --pattern can only be used when input_path is a folder")
        if path.suffix.lower() != ".txt":
            raise ValueError(f"Input file must be a .txt file: {path}")
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Input path not found: {path}")

    if file_name and pattern:
        raise ValueError("Use either --file or --pattern, not both")

    if file_name:
        selected = (path / file_name).resolve()
        if not selected.is_file():
            raise FileNotFoundError(f"TXT file not found in folder: {selected}")
        if selected.suffix.lower() != ".txt":
            raise ValueError(f"--file must select a .txt file: {selected}")
        return [selected]

    if pattern:
        files = sorted(p for p in path.glob(pattern) if p.is_file() and p.suffix.lower() == ".txt")
        if not files:
            raise FileNotFoundError(f"No .txt files matched pattern {pattern!r} in {path}")
        return files

    files = sorted(p for p in path.glob("*.txt") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No .txt files found in folder: {path}")
    return files


def parse_true_mu_from_filename(path: Path) -> int | None:
    """Parse true MU from names like 500_70w_50_225000_1.txt."""
    parts = path.stem.split("_")
    if len(parts) < 3:
        return None
    try:
        return int(parts[2])
    except ValueError:
        return None


def choose_d1(path: Path, args: argparse.Namespace) -> tuple[int, int | None]:
    true_mu = parse_true_mu_from_filename(path)
    if args.D1 is not None:
        return args.D1, true_mu
    if true_mu is None:
        return args.default_D1, true_mu
    return max(1, true_mu - args.auto_start_below), true_mu


def result_json_path(work_dir: Path, txt_path: Path) -> Path:
    return work_dir / f"run_stairfit_{txt_path.stem}.json"


def is_completed_json(json_path: Path) -> bool:
    if not json_path.is_file():
        return False
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return data.get("status") == "ok"
    except (json.JSONDecodeError, OSError):
        return False


def matlab_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def resolve_stairfit_dir(path_arg: Path | None) -> Path:
    candidates: list[Path] = []
    if path_arg is not None:
        candidates.append(path_arg)

    env_path = os.environ.get("STAIRFIT_M_DIR")
    if env_path:
        candidates.append(Path(env_path))

    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    candidates.extend(
        [
            cwd,
            script_dir,
            cwd / "StairFit",
            cwd / "matlab" / "StairFit",
            cwd / "MATLAB" / "StairFit",
            script_dir / "StairFit",
            script_dir / "matlab" / "StairFit",
            script_dir / "MATLAB" / "StairFit",
        ]
    )

    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if (path / "StairFit.m").is_file():
            return path

    checked = ", ".join(str(item.expanduser()) for item in candidates)
    raise FileNotFoundError(
        "Could not find StairFit.m. Put StairFit.m near this script, use --stairfit-dir, "
        f"or set STAIRFIT_M_DIR. Checked: {checked}"
    )


def resolve_matlab_cmd(matlab_cmd: str) -> str:
    """Return an executable MATLAB path (PATH, then ~/matlab/bin/matlab)."""
    if Path(matlab_cmd).is_file():
        return str(Path(matlab_cmd).resolve())
    found = shutil.which(matlab_cmd)
    if found:
        return found
    for candidate in (
        Path.home() / "matlab" / "bin" / "matlab",
        Path("/usr/local/MATLAB/R2024a/bin/matlab"),
        Path("/usr/local/MATLAB/R2024b/bin/matlab"),
    ):
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(
        f"MATLAB command {matlab_cmd!r} was not found. "
        "Add matlab to PATH, install under ~/matlab, or pass --matlab-cmd /full/path/to/matlab."
    )


def ensure_matlab_available(matlab_cmd: str) -> None:
    resolve_matlab_cmd(matlab_cmd)


def write_matlab_wrapper(
    *,
    script_path: Path,
    input_path: Path,
    output_json: Path,
    stairfit_dir: Path,
    d1: int,
    d2: int,
    inc: int,
    th: float,
    true_mu: int | None,
    show_plot: bool,
    cmap_polarity: str,
) -> None:
    true_mu_expr = "[]" if true_mu is None else str(true_mu)
    figure_expr = "'on'" if show_plot else "'off'"

    script = f"""
try
    addpath({matlab_string(stairfit_dir)});
    set(0, 'DefaultFigureVisible', {figure_expr});

    data = readmatrix({matlab_string(input_path)}, 'FileType', 'text');

    if strcmp({matlab_string(cmap_polarity)}, 'invert')
        data(:,2) = -data(:,2);
    elseif strcmp({matlab_string(cmap_polarity)}, 'positive')
        data_for_polarity = sortrows(data, 1);
        edge = max(1, min(10, floor(size(data_for_polarity,1) / 10)));
        low_response = median(data_for_polarity(1:edge, 2));
        high_response = median(data_for_polarity(end-edge+1:end, 2));
        if high_response < low_response
            data(:,2) = -data(:,2);
        end
    end

    [MUNE, STAIR] = StairFit(data, {d1}, {d2}, {inc}, {th:.17g});

    k = min({d2}, numel(STAIR.error));
    if k > 0
        last_batch_min_error = min(STAIR.error(end-k+1:end));
    else
        last_batch_min_error = [];
    end

    result = struct();
    result.status = 'ok';
    result.file = {matlab_string(input_path.name)};
    result.path = {matlab_string(input_path)};
    result.rows = size(data, 1);
    result.true_mu = {true_mu_expr};
    result.d1_used = {d1};
    result.mune = MUNE.number;
    result.runtime_s = MUNE.runtime;
    result.candidate_fits = numel(STAIR.error);
    result.last_batch_min_error = last_batch_min_error;
    result.stair = reshape(MUNE.stair, 1, []);
    result.point = reshape(MUNE.point, 1, []);

    fid = fopen({matlab_string(output_json)}, 'w');
    fwrite(fid, jsonencode(result), 'char');
    fclose(fid);
catch ME
    result = struct();
    result.status = 'error';
    result.file = {matlab_string(input_path.name)};
    result.path = {matlab_string(input_path)};
    result.message = getReport(ME, 'extended', 'hyperlinks', 'off');
    fid = fopen({matlab_string(output_json)}, 'w');
    fwrite(fid, jsonencode(result), 'char');
    fclose(fid);
    exit(1);
end
exit(0);
"""
    script_path.write_text(script, encoding="utf-8")


def read_log_tail(log_path: Path, *, max_lines: int = 40) -> str:
    if not log_path.is_file():
        return "(no log file)"
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def run_matlab_script(
    matlab_cmd: str,
    script_path: Path,
    verbose: bool,
    *,
    log_path: Path,
    timeout_s: int | None,
) -> None:
    """Run MATLAB without capture_output (avoids pipe deadlock when MATLAB is chatty)."""
    matlab_exe = resolve_matlab_cmd(matlab_cmd)
    command = [matlab_exe, "-batch", f"run({matlab_string(script_path)})"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"  MATLAB command: {' '.join(command)}", flush=True)
    print(f"  MATLAB log: {log_path}", flush=True)

    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
            )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"MATLAB timed out after {timeout_s}s (log: {log_path}). "
            "Increase --matlab-timeout or rerun this file with --file."
        ) from None

    if completed.returncode != 0:
        raise RuntimeError(
            f"MATLAB failed with exit code {completed.returncode}\n"
            f"Log: {log_path}\n--- tail ---\n{read_log_tail(log_path)}"
        )


def as_list(value: object) -> list[float]:
    if value is None or value == []:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(item) for item in value]  # type: ignore[arg-type]


def nullable_float(value: object) -> float | None:
    if value is None or value == []:
        return None
    return float(value)  # type: ignore[arg-type]


def load_result_from_json(json_path: Path, txt_path: Path) -> dict[str, object]:
    """从已完成的 checkpoint json 恢复与 run_one_file 相同结构的结果。"""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if data.get("status") != "ok":
        raise RuntimeError(data.get("message", f"checkpoint not ok: {json_path}"))

    true_mu = parse_true_mu_from_filename(txt_path)
    if data.get("true_mu") not in (None, [], ""):
        try:
            true_mu = int(data["true_mu"])
        except (TypeError, ValueError):
            pass

    mune = int(data.get("mune", 0))
    stair_values = as_list(data.get("stair"))
    point_values = as_list(data.get("point"))
    last_error = nullable_float(data.get("last_batch_min_error"))

    return {
        "file": txt_path.name,
        "path": str(txt_path.resolve()),
        "rows": int(data.get("rows", 0)),
        "true_mu": true_mu,
        "d1_used": int(data.get("d1_used", 0)),
        "mune": mune,
        "abs_err": abs(mune - true_mu) if true_mu is not None else None,
        "runtime_s": float(data.get("runtime_s", 0)),
        "candidate_fits": int(data.get("candidate_fits", 0)),
        "last_batch_min_error": last_error,
        "stair": stair_values,
        "point": point_values,
    }


def run_one_file(path: Path, args: argparse.Namespace, *, show_plot: bool) -> dict[str, object]:
    file_start = perf_counter()
    d1, true_mu = choose_d1(path, args)
    print(
        f"  true_mu={true_mu if true_mu is not None else 'N/A'}, "
        f"D1={d1}, D2={args.D2}, inc={args.inc}, th={args.th}",
        flush=True,
    )

    script_path = args.work_dir / f"run_stairfit_{path.stem}.m"
    output_json = result_json_path(args.work_dir, path)
    write_matlab_wrapper(
        script_path=script_path,
        input_path=path,
        output_json=output_json,
        stairfit_dir=args.stairfit_dir,
        d1=d1,
        d2=args.D2,
        inc=args.inc,
        th=args.th,
        true_mu=true_mu,
        show_plot=show_plot,
        cmap_polarity=args.cmap_polarity,
    )
    matlab_log = args.work_dir / f"{script_path.stem}.matlab.log"
    run_matlab_script(
        args.matlab_cmd,
        script_path,
        args.verbose,
        log_path=matlab_log,
        timeout_s=args.matlab_timeout,
    )

    if not output_json.is_file():
        raise FileNotFoundError(f"MATLAB did not create result JSON: {output_json}")

    result = json.loads(output_json.read_text(encoding="utf-8"))
    if result.get("status") != "ok":
        raise RuntimeError(result.get("message", "MATLAB StairFit failed"))

    stair_values = as_list(result.get("stair"))
    point_values = as_list(result.get("point"))
    last_error = nullable_float(result.get("last_batch_min_error"))

    print("=" * 72)
    print(f"Input: {path}")
    print(f"Rows: {result.get('rows')}")
    print(f"True MU from filename: {true_mu if true_mu is not None else 'N/A'}")
    print(f"D1 used: {d1}")
    print(f"MUNE: {result.get('mune')}")
    print(f"Runtime (s): {float(result.get('runtime_s', 0)):.3f}")
    print(f"Candidate fits: {result.get('candidate_fits')}")
    if last_error is not None:
        print(f"Last batch min error: {last_error:.6f}")
    if args.verbose:
        print(f"Stair heights: {stair_values}")
        print(f"Activation points: {point_values}")
    print(f"File finished in {perf_counter() - file_start:.3f} s", flush=True)

    return {
        "file": path.name,
        "path": str(path),
        "rows": int(result.get("rows", 0)),
        "true_mu": true_mu,
        "d1_used": d1,
        "mune": int(result.get("mune", 0)),
        "abs_err": abs(int(result.get("mune", 0)) - true_mu) if true_mu is not None else None,
        "runtime_s": float(result.get("runtime_s", 0)),
        "candidate_fits": int(result.get("candidate_fits", 0)),
        "last_batch_min_error": last_error,
        "stair": stair_values,
        "point": point_values,
    }


def autosize_columns(ws) -> None:
    for column_cells in ws.columns:
        max_len = 0
        col_letter = get_column_letter(column_cells[0].column)
        for cell in column_cells:
            value = "" if cell.value is None else str(cell.value)
            max_len = max(max_len, len(value))
        ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 60)


def style_sheet(ws) -> None:
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    autosize_columns(ws)


def save_results_excel(results: list[dict[str, object]], output_path: Path) -> None:
    if not results:
        return

    wb = Workbook()

    summary = wb.active
    summary.title = "Summary"
    summary.append(
        [
            "File",
            "Rows",
            "True MU",
            "D1 Used",
            "MUNE",
            "Abs Error",
            "Runtime (s)",
            "Candidate Fits",
            "Last Batch Min Error",
            "Full Path",
        ]
    )
    for result in results:
        summary.append(
            [
                result["file"],
                result["rows"],
                result["true_mu"],
                result["d1_used"],
                result["mune"],
                result["abs_err"],
                result["runtime_s"],
                result["candidate_fits"],
                result["last_batch_min_error"],
                result["path"],
            ]
        )

    stairs = wb.create_sheet("StairHeights")
    max_stair_len = max(len(result["stair"]) for result in results)
    stairs.append(["File", "True MU", "MUNE"] + [f"Stair {i + 1}" for i in range(max_stair_len)])
    for result in results:
        values = list(result["stair"])
        stairs.append(
            [result["file"], result["true_mu"], result["mune"]]
            + values
            + [None] * (max_stair_len - len(values))
        )

    points = wb.create_sheet("ActivationPoints")
    max_point_len = max(len(result["point"]) for result in results)
    points.append(["File", "True MU", "MUNE"] + [f"Point {i + 1}" for i in range(max_point_len)])
    for result in results:
        values = list(result["point"])
        points.append(
            [result["file"], result["true_mu"], result["mune"]]
            + values
            + [None] * (max_point_len - len(values))
        )

    for ws in wb.worksheets:
        style_sheet(ws)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Call MATLAB StairFit.m on data_msf TXT files and save Excel results."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        default=Path("data_msf"),
        help="Folder of .txt files or a single .txt file path. Default: data_msf",
    )
    parser.add_argument("--file", help="Specific TXT file name inside input_path folder.")
    parser.add_argument("--pattern", help='Glob pattern inside input_path folder, e.g. "*.txt".')
    parser.add_argument("--limit", type=int, default=None, help="Process at most N files after sorting.")
    parser.add_argument("--D1", type=int, default=None, help="Override initial MUNE D1.")
    parser.add_argument(
        "--auto-start-below",
        type=int,
        default=20,
        help="If true MU is parsed from filename, use D1=max(1, true_mu-this). Default: 20.",
    )
    parser.add_argument(
        "--default-D1",
        type=int,
        default=20,
        help="Fallback D1 when true MU cannot be parsed. Default: 20.",
    )
    parser.add_argument("--D2", type=int, default=8, help="Parallel models per MATLAB StairFit batch.")
    parser.add_argument("--inc", type=int, default=2, help="MUNE increment/resolution.")
    parser.add_argument("--th", type=float, default=0.015, help="Fitting error threshold.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/res_sf_70w"),
        help="Output directory. Default: results/res_sf_70w",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Excel file path. Overrides --output-dir.",
    )
    parser.add_argument(
        "--stairfit-dir",
        type=Path,
        help="Folder containing original MATLAB StairFit.m.",
    )
    parser.add_argument(
        "--matlab-cmd",
        default="matlab",
        help="MATLAB executable command. Default: matlab.",
    )
    parser.add_argument(
        "--matlab-timeout",
        type=int,
        default=10800,
        metavar="SECONDS",
        help="Kill MATLAB if a single file exceeds this wall time. Default: 10800 (3h). Use 0 to disable.",
    )
    parser.add_argument(
        "--cmap-polarity",
        choices=["positive", "invert", "as_is"],
        default="as_is",
        help="Optional CMAP preprocessing before MATLAB. Default: as_is.",
    )
    parser.add_argument("--plot", action="store_true", help="Show MATLAB figure for single-file mode.")
    parser.add_argument("--verbose", action="store_true", help="Print MATLAB command/output and arrays.")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing _matlab_runs/*.json and rerun MATLAB for every file.",
    )
    parser.add_argument(
        "--no-checkpoint-excel",
        action="store_true",
        help="Only write Excel at the end (default: update Excel after each file for resume safety).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="On MATLAB failure/timeout, log and continue with remaining files instead of aborting.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    paths = resolve_input_files(args.input_path, args.file, args.pattern)
    if args.limit is not None:
        paths = paths[: max(0, args.limit)]
    if not paths:
        print("No files to process.", flush=True)
        return
    if len(paths) > 1 and args.plot:
        raise ValueError("--plot is only supported when exactly one file is selected")

    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else (args.output_dir.expanduser().resolve() / "stairfit_results.xlsx")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    args.stairfit_dir = resolve_stairfit_dir(args.stairfit_dir)
    args.matlab_cmd = resolve_matlab_cmd(args.matlab_cmd)
    args.matlab_timeout = None if args.matlab_timeout == 0 else args.matlab_timeout
    args.work_dir = output_path.parent / "_matlab_runs"
    args.work_dir.mkdir(parents=True, exist_ok=True)

    total_start = perf_counter()
    print(f"Selected files: {len(paths)}", flush=True)
    print(f"Output Excel: {output_path}", flush=True)
    print(f"MATLAB: {args.matlab_cmd}", flush=True)
    print(f"MATLAB StairFit dir: {args.stairfit_dir}", flush=True)
    print(f"Checkpoint dir: {args.work_dir}", flush=True)

    if not args.no_resume:
        n_done = sum(1 for p in paths if is_completed_json(result_json_path(args.work_dir, p)))
        print(f"Resume: {n_done}/{len(paths)} already completed (will skip MATLAB)", flush=True)
    else:
        print("Resume: disabled (--no-resume)", flush=True)

    results: list[dict[str, object]] = []
    checkpoint_excel = not args.no_checkpoint_excel

    for index, path in enumerate(paths, start=1):
        json_path = result_json_path(args.work_dir, path)
        if not args.no_resume and is_completed_json(json_path):
            print(f"[{index}/{len(paths)}] skip (resume) {path.name}", flush=True)
            results.append(load_result_from_json(json_path, path))
        else:
            if json_path.is_file() and not args.no_resume:
                print(f"[{index}/{len(paths)}] rerun (incomplete checkpoint) {path.name}", flush=True)
            else:
                print(f"[{index}/{len(paths)}] Running {path.name} ...", flush=True)
            try:
                results.append(run_one_file(path, args, show_plot=args.plot))
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                print(f"  ERROR (skipped): {exc}", flush=True)
                fail_log = args.work_dir / "failed_files.log"
                with open(fail_log, "a", encoding="utf-8") as fh:
                    fh.write(f"{path.name}\t{exc}\n")

        if checkpoint_excel and results:
            save_results_excel(results, output_path)
            print(f"  checkpoint Excel updated ({len(results)} rows)", flush=True)

    save_results_excel(results, output_path)
    print("=" * 72)
    print(f"Excel saved: {output_path} ({len(results)} files)", flush=True)
    print(f"Total runtime: {perf_counter() - total_start:.3f} s", flush=True)


if __name__ == "__main__":
    main()
