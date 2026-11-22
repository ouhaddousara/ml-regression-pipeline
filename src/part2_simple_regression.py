"""Part 2 - Simple linear regression (Experience -> Salary).

Assignment items covered
  1. train a LinearRegression with scikit-learn
  2. predict the test set
  3. plot the regression result
  4. evaluate with MSE, RMSE, MAE and interpret (interpretation lives in the report)

Run from the repository root:  python src/part2_simple_regression.py
"""
from __future__ import annotations

from utils import (  # utils first: it selects the Agg backend
    RANDOM_STATE,
    RES_DIR,
    TEST_SIZE,
    baseline_rmse,
    df_to_latex,
    ensure_dirs,
    load_dataset,
    metric_macros,
    metrics_table_tex,
    print_metrics,
    regression_metrics,
    save_fig,
    save_json,
    scale_macros,
    section,
    write_tex,
    fmt_num,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

FEATURE, TARGET = "YearsExperience", "Salary"


def main() -> None:
    ensure_dirs()
    pd.set_option("display.width", 200)

    df = load_dataset("salary")
    n_before = len(df)
    df = df[[FEATURE, TARGET]].apply(pd.to_numeric, errors="coerce").dropna()
    print(f"Rows kept after numeric conversion / NaN removal: {len(df)} / {n_before}")

    # ------------------------------------------------------------ 1. training
    section("Item 1 - train/test split and LinearRegression fit")
    X, y = df[[FEATURE]], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"train = {len(X_train)} rows, test = {len(X_test)} rows "
          f"(test_size={TEST_SIZE}, random_state={RANDOM_STATE})")
    model = LinearRegression().fit(X_train, y_train)
    slope, intercept = float(model.coef_[0]), float(model.intercept_)
    print(f"Model: {TARGET} = {intercept:.4f} + {slope:.4f} x {FEATURE}")

    # ---------------------------------------------------------- 2. prediction
    section("Item 2 - predictions on the test set")
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    pred_df = pd.DataFrame(
        {
            FEATURE: X_test[FEATURE].values,
            "Salaire réel": y_test.values,
            "Salaire prédit": pred_test,
            "Erreur (réel - prédit)": y_test.values - pred_test,
        }
    ).sort_values(FEATURE)
    print(pred_df.to_string(index=False))
    pred_df.to_csv(RES_DIR / "part2_predictions.csv", index=False)
    write_tex(
        "part2_predictions.tex",
        {},
        df_to_latex(pred_df, "Pr\\'edictions sur le jeu de test (Partie 2)",
                    "tab:p2_predictions", index=False),
    )

    # -------------------------------------------------------------- 3. figure
    section("Item 3 - regression plot")
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    ax.scatter(X_train[FEATURE], y_train, color="tab:blue", edgecolor="black",
               s=50, label=f"Train (n = {len(X_train)})")
    ax.scatter(X_test[FEATURE], y_test, color="tab:orange", edgecolor="black",
               marker="s", s=65, label=f"Test (n = {len(X_test)})")
    grid = pd.DataFrame({FEATURE: np.linspace(df[FEATURE].min(), df[FEATURE].max(), 200)})
    ax.plot(grid[FEATURE], model.predict(grid), color="tab:red", linewidth=2.2,
            label=f"Droite de régression : y = {slope:.1f}·x {intercept:+.1f}")
    ax.vlines(X_test[FEATURE], y_test, pred_test, colors="gray", linestyles="dotted",
              label="Résidus (jeu de test)")
    ax.set_title("Régression linéaire simple : salaire vs expérience")
    ax.set_xlabel("Années d'expérience (YearsExperience)")
    ax.set_ylabel("Salaire (Salary)")
    ax.legend(loc="upper left")
    save_fig(fig, "p2_regression_plot.png")

    # ---------------------------------------------------------- 4. evaluation
    section("Item 4 - evaluation (MSE, RMSE, MAE)")
    m_train = regression_metrics(y_train, pred_train)
    m_test = regression_metrics(y_test, pred_test)
    base = baseline_rmse(y_train, y_test)
    print_metrics("TRAIN", m_train)
    print_metrics("TEST", m_test)
    print(f"Baseline RMSE (always predict the train mean): {base:.6g}")
    print(f"Target mean = {y.mean():.6g}, target std = {y.std():.6g}")
    print(f"RMSE(test) / mean(target) = {100 * m_test['rmse'] / y.mean():.2f} %")
    print("Overfitting check: compare TRAIN and TEST RMSE / R2 above.")

    save_json(
        "part2_metrics.json",
        {
            "dataset": "Experience / Salary",
            "feature": FEATURE,
            "target": TARGET,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "slope": slope,
            "intercept": intercept,
            "train": m_train,
            "test": m_test,
            "baseline_rmse": base,
            "target_mean": float(y.mean()),
            "target_std": float(y.std()),
        },
    )

    macros = {
        "PartTwoSlope": fmt_num(slope),
        "PartTwoIntercept": fmt_num(intercept),
    }
    macros.update(metric_macros("PartTwo", m_test, m_train, y_mean=float(y.mean())))
    macros.update(scale_macros("PartTwo", y, y_train, y_test))
    table = metrics_table_tex(
        [("Train", m_train), ("Test", m_test)],
        "M\\'etriques de la r\\'egression lin\\'eaire simple (unit\\'es du salaire)",
        "tab:p2_metrics",
    )
    write_tex("part2_metrics.tex", macros, table)
    print("\nPart 2 finished.")


if __name__ == "__main__":
    main()
