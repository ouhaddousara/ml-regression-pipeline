"""Shared helpers for the Atelier 1 regression project.

Conventions used by every script
--------------------------------
* Matplotlib backend: Agg (no display needed).
* RANDOM_STATE = 42 and TEST_SIZE = 0.2 everywhere (single convention).
* Figures  -> report/figures/*.png (200 dpi).
* Metrics  -> results/*.json (+ some CSV).
* LaTeX    -> report/generated/*.tex: booktabs tables and \\providecommand/
  \\renewcommand macros, so that report/main.tex pulls the real numbers via \\input.

Usage as a CLI (dataset download):  python src/utils.py fetch
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # select the non-interactive backend before pyplot is used
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# --------------------------------------------------------------------------- #
# Global conventions
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.2
DPI = 200

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FIG_DIR = ROOT / "report" / "figures"
GEN_DIR = ROOT / "report" / "generated"
SHOT_DIR = ROOT / "report" / "screenshots"
RES_DIR = ROOT / "results"
LOG_DIR = RES_DIR / "logs"

# Expected columns are ASSUMPTIONS: they are checked at load time, never trusted.
DATASETS = {
    "salary": {
        "names": ["Salary_dataset.csv", "Salary_Data.csv"],
        "hint": "salary",
        "expected": ["YearsExperience", "Salary"],
    },
    "insurance": {
        "names": ["insurance.csv"],
        "hint": "insurance",
        "expected": ["age", "sex", "bmi", "children", "smoker", "region", "charges"],
    },
    "china_gdp": {
        "names": ["china_gdp.csv"],
        "hint": "gdp",
        "expected": ["Year", "Value"],
    },
}

KAGGLE_SLUGS = {
    "salary": "rohankayan/years-of-experience-and-salary-dataset",
    "insurance": "sinaasappel/tutorial-multiple-regression",
}

CHINA_GDP_URLS = [
    # URL given in the assignment
    "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/"
    "CognitiveClass/ML0101ENv3/labs/china_gdp.csv",
    # Possible IBM mirror (to verify; the manual download remains the fallback)
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/china_gdp.csv",
]

plt.rcParams.update(
    {
        "figure.dpi": 100,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# --------------------------------------------------------------------------- #
# Filesystem / printing helpers
# --------------------------------------------------------------------------- #
def ensure_dirs() -> None:
    """Create every output directory used by the scripts."""
    for d in (RAW_DIR, FIG_DIR, GEN_DIR, SHOT_DIR, RES_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    """Print a visible banner in the terminal log."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def rel(path: Path) -> str:
    """Path relative to the repository root, for readable logs."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def save_fig(fig, filename: str) -> Path:
    """Save a figure to report/figures/ (PNG, 200 dpi) and close it."""
    ensure_dirs()
    path = FIG_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] saved {rel(path)}")
    return path


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def save_json(filename: str, obj) -> Path:
    """Write a JSON file into results/."""
    ensure_dirs()
    path = RES_DIR / filename
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=_json_default)
    print(f"[results] saved {rel(path)}")
    return path


# --------------------------------------------------------------------------- #
# Dataset loading (defensive)
# --------------------------------------------------------------------------- #
def require_columns(df: pd.DataFrame, expected: list[str], name: str) -> None:
    """Print df.columns and abort with a clear message if a column is missing."""
    print(f"[{name}] columns found: {list(df.columns)}")
    missing = [c for c in expected if c not in df.columns]
    if missing:
        sys.exit(
            f"ERROR [{name}] expected column(s) {missing} not found. "
            f"Columns in the file: {list(df.columns)}. "
            "The expected names are assumptions: check data/raw/ and adapt the "
            "'expected' list in DATASETS (src/utils.py) if the dataset differs."
        )


def find_dataset(key: str) -> Path | None:
    """Locate the CSV of a dataset in data/raw/ (exact name first, then a name hint)."""
    spec = DATASETS[key]
    csvs = sorted(RAW_DIR.rglob("*.csv")) if RAW_DIR.exists() else []
    wanted = [n.lower() for n in spec["names"]]
    for p in csvs:
        if p.name.lower() in wanted:
            return p
    for p in csvs:
        if spec["hint"] in p.name.lower():
            return p
    return None


def load_dataset(key: str) -> pd.DataFrame:
    """Load a dataset, drop index-like 'Unnamed' columns, validate expected columns."""
    spec = DATASETS[key]
    path = find_dataset(key)
    if path is None:
        found = [rel(p) for p in sorted(RAW_DIR.rglob("*.csv"))] if RAW_DIR.exists() else []
        sys.exit(
            f"ERROR: no CSV found for dataset '{key}' in {rel(RAW_DIR)}. "
            f"Expected one of {spec['names']}. CSV files found: {found}. "
            "Run `make data` or place the file manually (see data/README.md)."
        )
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).str.startswith("Unnamed")]
    print(f"[{key}] loaded {rel(path)} -> shape {df.shape}")
    require_columns(df, spec["expected"], key)
    return df


# --------------------------------------------------------------------------- #
# Dataset download (python src/utils.py fetch)
# --------------------------------------------------------------------------- #
def _download_china_gdp() -> bool:
    target = RAW_DIR / "china_gdp.csv"
    for url in CHINA_GDP_URLS:
        try:
            print(f"[fetch] trying {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if b"Year" not in data[:300]:
                print("[fetch] response does not look like the expected CSV, skipped")
                continue
            target.write_bytes(data)
            print(f"[fetch] saved {rel(target)}")
            return True
        except Exception as exc:  # network errors, HTTP errors, ...
            print(f"[fetch] failed: {exc}")
    return False


def _download_kaggle(slug: str) -> bool:
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        print(
            f"[fetch] KAGGLE_USERNAME / KAGGLE_KEY not set in the environment: "
            f"cannot download '{slug}' automatically."
        )
        return False
    exe = shutil.which("kaggle") or str(Path(sys.executable).parent / "kaggle")
    cmd = [exe, "datasets", "download", "-d", slug, "-p", str(RAW_DIR), "--unzip"]
    try:
        print(f"[fetch] kaggle datasets download -d {slug}")
        subprocess.run(cmd, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[fetch] Kaggle download failed for '{slug}': {exc}")
        return False


def fetch_data() -> int:
    """Download missing datasets. Returns 0 if all are present afterwards, else 1."""
    ensure_dirs()
    for key in DATASETS:
        found = find_dataset(key)
        if found is not None:
            print(f"[fetch] '{key}' already present: {rel(found)}")
            continue
        if key == "china_gdp":
            _download_china_gdp()
        else:
            _download_kaggle(KAGGLE_SLUGS[key])
    missing = [k for k in DATASETS if find_dataset(k) is None]
    if missing:
        print(
            "\n[fetch] Still missing: " + ", ".join(missing) + "\n"
            "Manual fallback: download the files in your browser and place them in "
            f"{rel(RAW_DIR)}/ (see data/README.md for the expected file names)."
        )
        return 1
    print("\n[fetch] All datasets are present in data/raw/.")
    return 0


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def regression_metrics(y_true, y_pred) -> dict:
    """MSE, RMSE, MAE (requested by the assignment) and R^2 (supplementary)."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def baseline_rmse(y_train, y_test) -> float:
    """RMSE of the naive model that always predicts the training mean."""
    y_test = np.asarray(y_test, dtype=float)
    return float(np.sqrt(np.mean((y_test - float(np.mean(y_train))) ** 2)))


def print_metrics(label: str, m: dict) -> None:
    print(
        f"{label:<28} MSE={m['mse']:.6g}  RMSE={m['rmse']:.6g}  "
        f"MAE={m['mae']:.6g}  R2={m['r2']:.4f}"
    )


# --------------------------------------------------------------------------- #
# LaTeX generation helpers
# --------------------------------------------------------------------------- #
def tex_escape(value) -> str:
    """Escape the LaTeX special characters that appear in our cell values."""
    s = str(value)
    for a, b in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_")):
        s = s.replace(a, b)
    return s


def fmt_num(x, digits: int = 4) -> str:
    """Format a number for LaTeX (thin-space thousands, scientific for extremes)."""
    if x is None or isinstance(x, (bool, np.bool_)):
        return "--"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}".replace(",", r"\,")
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return "--"
    ax = abs(x)
    if ax < 1e6 and x.is_integer():
        return f"{int(x):,}".replace(",", r"\,")
    if x == 0:
        return "0"
    if ax >= 1e6 or ax < 1e-3:
        mant, exp = f"{x:.{digits - 1}e}".split("e")
        # \ensuremath works in text mode AND inside an existing math environment
        return f"\\ensuremath{{{mant}\\times10^{{{int(exp)}}}}}"
    if ax >= 1000:
        return f"{x:,.2f}".replace(",", r"\,")
    return f"{x:.{digits}f}"


def df_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    index: bool = True,
    index_header: str = "",
    resize: bool = False,
) -> str:
    """DataFrame -> booktabs table. Headers are raw LaTeX (caller makes them safe);
    index labels and string cells are escaped; numbers go through fmt_num."""
    cols = list(df.columns)
    align = ("l" if index else "") + "".join(
        "r" if pd.api.types.is_numeric_dtype(df[c]) else "l" for c in cols
    )
    header = ([index_header] if index else []) + [str(c) for c in cols]
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}"]
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines += [rf"\begin{{tabular}}{{{align}}}", r"\toprule",
              " & ".join(header) + r" \\", r"\midrule"]
    for idx, row in df.iterrows():
        cells = [tex_escape(idx)] if index else []
        for c in cols:
            v = row[c]
            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                cells.append(fmt_num(v))
            else:
                cells.append(tex_escape(v))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if resize:
        lines.append("}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def metrics_table_tex(rows: list[tuple[str, dict]], caption: str, label: str) -> str:
    """Booktabs table with MSE, RMSE, MAE (and R^2) for each (name, metrics) row."""
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Jeu / mod\`ele & MSE & RMSE & MAE & $R^2$ \\", r"\midrule"]
    for name, m in rows:
        lines.append(
            f"{tex_escape(name)} & {fmt_num(m['mse'])} & {fmt_num(m['rmse'])} & "
            f"{fmt_num(m['mae'])} & {fmt_num(m['r2'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def write_tex(filename: str, macros: dict[str, str], body: str = "") -> Path:
    """Write report/generated/<filename>: macros first, then an optional LaTeX body.

    Macro names must contain letters only (LaTeX restriction). \\providecommand +
    \\renewcommand makes the file safe whether or not main.tex predefined the macro.
    """
    ensure_dirs()
    lines = ["% Auto-generated by the Python scripts - do not edit by hand."]
    for name, value in macros.items():
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"Invalid LaTeX macro name: {name!r} (letters only)")
        lines.append(f"\\providecommand{{\\{name}}}{{}}")
        lines.append(f"\\renewcommand{{\\{name}}}{{{value}}}")
    path = GEN_DIR / filename
    path.write_text("\n".join(lines) + "\n" + body, encoding="utf-8")
    print(f"[latex] saved {rel(path)}")
    return path


def metric_macros(prefix: str, test: dict, train: dict | None = None,
                  y_mean: float | None = None) -> dict[str, str]:
    """Macros <prefix>MSE/RMSE/MAE/RTwo (+ Train* and RMSEPct/MAEPct when available)."""
    macros = {
        f"{prefix}MSE": fmt_num(test["mse"]),
        f"{prefix}RMSE": fmt_num(test["rmse"]),
        f"{prefix}MAE": fmt_num(test["mae"]),
        f"{prefix}RTwo": fmt_num(test["r2"]),
    }
    if train is not None:
        macros.update(
            {
                f"{prefix}TrainMSE": fmt_num(train["mse"]),
                f"{prefix}TrainRMSE": fmt_num(train["rmse"]),
                f"{prefix}TrainMAE": fmt_num(train["mae"]),
                f"{prefix}TrainRTwo": fmt_num(train["r2"]),
            }
        )
    if y_mean:
        macros[f"{prefix}RMSEPct"] = f"{100.0 * test['rmse'] / y_mean:.1f}\\,\\%"
        macros[f"{prefix}MAEPct"] = f"{100.0 * test['mae'] / y_mean:.1f}\\,\\%"
    return macros


def scale_macros(prefix: str, y_all, y_train, y_test) -> dict[str, str]:
    """Dataset-level macros used to interpret errors relative to the target scale."""
    return {
        f"{prefix}NTrain": fmt_num(len(y_train)),
        f"{prefix}NTest": fmt_num(len(y_test)),
        f"{prefix}TargetMean": fmt_num(float(np.mean(y_all))),
        f"{prefix}TargetStd": fmt_num(float(np.std(y_all, ddof=1))),
        f"{prefix}BaselineRMSE": fmt_num(baseline_rmse(y_train, y_test)),
    }


if __name__ == "__main__":
    if sys.argv[1:2] == ["fetch"]:
        sys.exit(fetch_data())
    print("Usage: python src/utils.py fetch")
    sys.exit(2)
