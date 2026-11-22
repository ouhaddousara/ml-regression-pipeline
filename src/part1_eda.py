"""Part 1 - Data exploration and visualisation (Salary + Insurance data sets).

Assignment items covered
  1. pandas exploration of the two data sets
  2. statistical summary (describe) of both data sets
  3. scatter plot Experience / Salary (the conclusion is written in the report)
  4. scatter_matrix of the Insurance features

Run from the repository root:  python src/part1_eda.py
"""
from __future__ import annotations

from utils import (  # utils first: it selects the Agg backend
    RES_DIR,
    df_to_latex,
    ensure_dirs,
    fmt_num,
    load_dataset,
    save_fig,
    save_json,
    section,
    write_tex,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from pandas.plotting import scatter_matrix

SAL_FEATURE, SAL_TARGET = "YearsExperience", "Salary"
INS_CATEGORICAL = ["sex", "smoker", "region"]
INS_NUMERIC = ["age", "bmi", "children", "charges"]


def explore(df: pd.DataFrame, name: str) -> dict:
    """Item 1: head / shape / dtypes / info / missing values / duplicates."""
    section(f"{name} - exploration")
    print("shape:", df.shape)
    print("\n-- head()")
    print(df.head())
    print("\n-- info()")
    df.info()
    print("\n-- missing values per column")
    print(df.isna().sum())
    duplicates = int(df.duplicated().sum())
    print("\nduplicated rows:", duplicates)
    print("\n-- unique values per column")
    print(df.nunique())
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_total": int(df.isna().sum().sum()),
        "duplicates": duplicates,
    }


def describe_numeric(df: pd.DataFrame, key: str, caption: str, label: str) -> None:
    """Item 2: statistical summary, printed, saved as CSV and as a LaTeX table."""
    section(f"{key} - describe() (numeric columns)")
    desc = df.describe().T
    print(desc)
    path = RES_DIR / f"part1_{key}_describe.csv"
    desc.to_csv(path)
    shown = desc.rename(columns={"25%": r"25\%", "50%": r"50\%", "75%": r"75\%"})
    write_tex(
        f"part1_{key}_describe.tex",
        {},
        df_to_latex(shown, caption, label, index=True, index_header="Variable", resize=True),
    )


def categorical_summary(df: pd.DataFrame) -> None:
    """Counts of each category of the Insurance categorical columns."""
    section("Insurance - categorical columns (value counts)")
    rows = []
    for col in INS_CATEGORICAL:
        counts = df[col].value_counts()
        print(f"\n{col}:\n{counts}")
        for category, n in counts.items():
            rows.append(
                {
                    "Variable": col,
                    "Modalité": category,
                    "Effectif": int(n),
                    "Part (\\%)": f"{100.0 * n / len(df):.1f}",
                }
            )
    write_tex(
        "part1_insurance_categorical.tex",
        {},
        df_to_latex(
            pd.DataFrame(rows),
            "Variables cat\\'egorielles du jeu Insurance",
            "tab:p1_insurance_categorical",
            index=False,
        ),
    )


def plot_salary_scatter(df: pd.DataFrame) -> float:
    """Item 3: scatter plot Experience / Salary with pandas + matplotlib."""
    r = float(df[SAL_FEATURE].corr(df[SAL_TARGET]))  # Pearson correlation
    print(f"\nPearson correlation {SAL_FEATURE} vs {SAL_TARGET}: {r:.4f}")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    df.plot.scatter(x=SAL_FEATURE, y=SAL_TARGET, ax=ax, color="tab:blue",
                    edgecolor="black", s=45)
    ax.legend([ax.collections[0]], [f"Observations (n = {len(df)})"],
              title=f"Pearson r = {r:.3f}", loc="upper left")
    ax.set_title("Nuage de points : salaire en fonction de l'expérience")
    ax.set_xlabel("Années d'expérience (YearsExperience)")
    ax.set_ylabel("Salaire (Salary)")
    save_fig(fig, "p1_salary_scatter.png")
    return r


def plot_insurance_scatter_matrix(df: pd.DataFrame) -> None:
    """Item 4: pandas scatter_matrix of the numeric Insurance features, coloured by smoker."""
    plot_df = df[INS_NUMERIC + ["smoker"]].dropna()
    is_smoker = plot_df["smoker"].astype(str).str.lower().isin(["yes", "y", "1", "true"])
    colors = np.where(is_smoker, "tab:red", "tab:blue")
    axes = scatter_matrix(
        plot_df[INS_NUMERIC],
        figsize=(11, 11),
        diagonal="hist",
        alpha=0.5,
        c=colors,
        hist_kwds={"bins": 25, "color": "gray"},
    )
    fig = axes[0, 0].get_figure()
    fig.suptitle("Scatter matrix des variables numériques du jeu Insurance", y=0.94, fontsize=14)
    fig.legend(
        handles=[Patch(color="tab:red", label="smoker = yes"),
                 Patch(color="tab:blue", label="smoker = no")],
        loc="upper right",
        bbox_to_anchor=(0.97, 0.93),
    )
    save_fig(fig, "p1_insurance_scatter_matrix.png")


def main() -> None:
    ensure_dirs()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    salary = load_dataset("salary")
    insurance = load_dataset("insurance")

    summary = {
        "salary": explore(salary, "Salary"),
        "insurance": explore(insurance, "Insurance"),
    }

    describe_numeric(salary, "salary", "Résumé statistique du jeu Experience / Salary",
                     "tab:p1_salary_describe")
    describe_numeric(insurance, "insurance", "Résumé statistique (variables numériques) du jeu Insurance",
                     "tab:p1_insurance_describe")
    categorical_summary(insurance)

    section("Item 3 - scatter plot Experience / Salary")
    pearson = plot_salary_scatter(salary)

    section("Item 4 - scatter_matrix of the Insurance features")
    plot_insurance_scatter_matrix(insurance)

    summary["salary_pearson_r"] = pearson
    save_json("part1_summary.json", summary)

    write_tex(
        "part1_stats.tex",
        {
            "PartOneSalaryRows": fmt_num(summary["salary"]["rows"]),
            "PartOneSalaryCols": fmt_num(summary["salary"]["cols"]),
            "PartOneSalaryMissing": fmt_num(summary["salary"]["missing_total"]),
            "PartOneInsuranceRows": fmt_num(summary["insurance"]["rows"]),
            "PartOneInsuranceCols": fmt_num(summary["insurance"]["cols"]),
            "PartOneInsuranceMissing": fmt_num(summary["insurance"]["missing_total"]),
            "PartOneInsuranceDuplicates": fmt_num(summary["insurance"]["duplicates"]),
            "PartOneSalaryCorr": f"{pearson:.4f}",
        },
    )
    print("\nPart 1 finished.")


if __name__ == "__main__":
    main()
