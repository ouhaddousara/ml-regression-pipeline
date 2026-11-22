"""Part 4 - Linear vs polynomial regression (China GDP: Year -> Value).

Assignment items covered
  1. train LinearRegression and polynomial regressions (degrees 2, 3, 4)
  2. predict the test set for every model
  3. plot both families of models
  4. evaluate with MSE, RMSE, MAE and interpret (interpretation lives in the report)

Model selection rule (explicit): among the polynomial degrees, keep the SMALLEST degree
whose test RMSE is within PARSIMONY_TOL (5 %) of the best test RMSE (parsimony rule).
Numerical stability: the Year feature is standardised inside each pipeline (scaler fitted
on the training split only) before PolynomialFeatures, otherwise Year**4 ~ 1e13.

Run from the repository root:  python src/part4_polynomial_regression.py
"""
from __future__ import annotations

from utils import (  # utils first: it selects the Agg backend
    RANDOM_STATE,
    RES_DIR,
    TEST_SIZE,
    baseline_rmse,
    df_to_latex,
    ensure_dirs,
    fmt_num,
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
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

FEATURE, TARGET = "Year", "Value"
DEGREES = [2, 3, 4]
PARSIMONY_TOL = 0.05
EXTRAP_YEARS = 10
SCALE = 1e12  # plots and the predictions table are expressed in units of 1e12
DEG_WORD = {2: "Two", 3: "Three", 4: "Four"}
COLORS = {"linear": "tab:gray", "poly2": "tab:green", "poly3": "tab:red", "poly4": "tab:purple"}


def model_label(name: str) -> str:
    return "Linéaire" if name == "linear" else f"Polynomiale d={name[-1]}"


def plot_comparison(models: dict, chosen: str, df: pd.DataFrame, X_train, y_train,
                    X_test, y_test) -> None:
    xmin, xmax = float(df[FEATURE].min()), float(df[FEATURE].max())
    x_obs = np.linspace(xmin, xmax, 300)
    x_ext = np.linspace(xmin, xmax + EXTRAP_YEARS, 400)

    def curve(mdl, xs):
        return mdl.predict(pd.DataFrame({FEATURE: xs})) / SCALE

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    ax.scatter(X_train[FEATURE], y_train / SCALE, color="tab:blue", edgecolor="black",
               s=30, label=f"Train (n = {len(X_train)})")
    ax.scatter(X_test[FEATURE], y_test / SCALE, color="tab:orange", edgecolor="black",
               marker="s", s=45, label=f"Test (n = {len(X_test)})")
    for name, mdl in models.items():
        lw = 3.0 if name == chosen else 1.6
        suffix = " (retenu)" if name == chosen else ""
        ax.plot(x_obs, curve(mdl, x_obs), color=COLORS[name], linewidth=lw,
                label=model_label(name) + suffix)
    ax.set_title("Ajustement sur la plage observée")
    ax.set_xlabel("Année (Year)")
    ax.set_ylabel("Valeur du PIB (Value, ×10¹² unités du CSV)")
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.scatter(df[FEATURE], df[TARGET] / SCALE, color="black", s=14, alpha=0.6,
               label="Données observées")
    for name, mdl in models.items():
        lw = 3.0 if name == chosen else 1.6
        ax.plot(x_ext, curve(mdl, x_ext), color=COLORS[name], linewidth=lw,
                label=model_label(name))
    ax.axvspan(xmax, xmax + EXTRAP_YEARS, color="gray", alpha=0.15,
               label=f"Extrapolation ({EXTRAP_YEARS} ans hors données)")
    ymax = float(df[TARGET].max() / SCALE)
    ax.set_ylim(min(0.0, float(df[TARGET].min() / SCALE)) - 0.05 * ymax, 3 * ymax)
    ax.set_title("Risque d'extrapolation (axe y borné)")
    ax.set_xlabel("Année (Year)")
    ax.set_ylabel("Valeur du PIB (Value, ×10¹² unités du CSV)")
    ax.legend(loc="upper left")

    fig.suptitle("Chine - PIB : régression linéaire vs polynomiale", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "p4_comparison.png")


def main() -> None:
    ensure_dirs()
    pd.set_option("display.width", 200)

    df = load_dataset("china_gdp")
    n_before = len(df)
    df = df[[FEATURE, TARGET]].apply(pd.to_numeric, errors="coerce").dropna()
    print(f"Rows kept after numeric conversion / NaN removal: {len(df)} / {n_before}")
    xmin, xmax = float(df[FEATURE].min()), float(df[FEATURE].max())
    print(f"Year range: {xmin:.0f} - {xmax:.0f}")

    # --------------------------------------------------- 1. training (4 models)
    section("Item 1 - train/test split and model fits")
    X, y = df[[FEATURE]], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"train = {len(X_train)} rows, test = {len(X_test)} rows "
          f"(test_size={TEST_SIZE}, random_state={RANDOM_STATE})")

    models = {"linear": make_pipeline(StandardScaler(), LinearRegression())}
    for d in DEGREES:
        models[f"poly{d}"] = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=d, include_bias=False),
            LinearRegression(),
        )
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)  # scaler is fitted on the TRAIN split only
        reg = mdl[-1]
        print(f"\n[{model_label(name)}] coefficients (in the standardised-Year space): "
              f"{np.array2string(reg.coef_, precision=6)}  intercept: {reg.intercept_:.6g}")

    # ---------------------------------------------------------- 2. predictions
    section("Item 2 - predictions on the test set")
    results = {}
    test_preds = {}
    for name, mdl in models.items():
        test_preds[name] = mdl.predict(X_test)
        results[name] = {
            "train": regression_metrics(y_train, mdl.predict(X_train)),
            "test": regression_metrics(y_test, test_preds[name]),
        }
    pred_df = pd.DataFrame({FEATURE: X_test[FEATURE].values, "Réel": y_test.values / SCALE})
    for name in models:
        pred_df[model_label(name)] = test_preds[name] / SCALE
    pred_df = pred_df.sort_values(FEATURE)
    print(pred_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    pred_df.to_csv(RES_DIR / "part4_test_predictions.csv", index=False)
    tex_pred = pred_df.copy()
    tex_pred[FEATURE] = tex_pred[FEATURE].astype(int).astype(str)
    tex_pred.columns = ["Ann\\'ee", "R\\'eel", "Lin\\'eaire", "Poly. $d=2$", "Poly. $d=3$", "Poly. $d=4$"]
    write_tex(
        "part4_predictions.tex",
        {},
        df_to_latex(tex_pred,
                    "Pr\\'edictions sur le jeu de test (en $10^{12}$ unit\\'es du CSV)",
                    "tab:p4_predictions", index=False, resize=True),
    )

    # ---------------------------------------------------------- 4. evaluation
    section("Item 4 - evaluation (MSE, RMSE, MAE) for every model")
    for name in models:
        print_metrics(f"{model_label(name)} TRAIN", results[name]["train"])
        print_metrics(f"{model_label(name)} TEST", results[name]["test"])
    base = baseline_rmse(y_train, y_test)
    print(f"Baseline RMSE (always predict the train mean): {base:.6g}")

    # ------------------------------------------------------ degree selection
    section("Degree selection (parsimony rule on the test RMSE)")
    best_rmse = min(results[f"poly{d}"]["test"]["rmse"] for d in DEGREES)
    eligible = [d for d in DEGREES
                if results[f"poly{d}"]["test"]["rmse"] <= (1 + PARSIMONY_TOL) * best_rmse]
    chosen_degree = min(eligible)
    chosen = f"poly{chosen_degree}"
    print(f"Best polynomial test RMSE = {best_rmse:.6g}; threshold (+{100 * PARSIMONY_TOL:.0f} %) "
          f"= {(1 + PARSIMONY_TOL) * best_rmse:.6g}")
    print(f"Eligible degrees: {eligible} -> CHOSEN DEGREE = {chosen_degree}")
    print("Caveat: the selection uses the test set, so the reported test error of the "
          "chosen model is slightly optimistic.")

    # --------------------------------------------------------- 3. figure
    section("Item 3 - comparison plot (and extrapolation risk)")
    plot_comparison(models, chosen, df, X_train, y_train, X_test, y_test)

    # ------------------------------------------------- extrapolation numbers
    future_year = xmax + EXTRAP_YEARS
    future = pd.DataFrame({FEATURE: [future_year]})
    extrap = {name: float(mdl.predict(future)[0]) for name, mdl in models.items()}
    print(f"\nPredictions at Year = {future_year:.0f} (outside the data, in 1e12 units):")
    for name, value in extrap.items():
        print(f"  {model_label(name):<18} {value / SCALE:.4f}")

    save_json(
        "part4_metrics.json",
        {
            "dataset": "China GDP",
            "feature": FEATURE,
            "target": TARGET,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "year_min": xmin,
            "year_max": xmax,
            "degrees_compared": DEGREES,
            "selection_rule": f"smallest degree with test RMSE <= (1 + {PARSIMONY_TOL}) x best polynomial test RMSE",
            "best_polynomial_test_rmse": best_rmse,
            "eligible_degrees": eligible,
            "chosen_degree": chosen_degree,
            "models": results,
            "baseline_rmse": base,
            "target_mean": float(y.mean()),
            "extrapolation_year": future_year,
            "extrapolation_predictions": extrap,
        },
    )

    macros = {
        "PartFourYearMin": f"{xmin:.0f}",
        "PartFourYearMax": f"{xmax:.0f}",
        "PartFourExtrapYear": f"{future_year:.0f}",
        "PartFourExtrapYears": str(EXTRAP_YEARS),
        "PartFourChosenDegree": str(chosen_degree),
        "PartFourBestTestRMSE": fmt_num(best_rmse),
        "PartFourTolPct": f"{100 * PARSIMONY_TOL:.0f}\\,\\%",
    }
    macros.update(scale_macros("PartFour", y, y_train, y_test))
    prefixes = {"linear": "PartFourLinear"}
    prefixes.update({f"poly{d}": f"PartFourPoly{DEG_WORD[d]}" for d in DEGREES})
    for name, prefix in prefixes.items():
        macros.update(metric_macros(prefix, results[name]["test"], results[name]["train"],
                                    y_mean=float(y.mean())))
        macros[f"{prefix}Extrap"] = fmt_num(extrap[name] / SCALE)
    macros.update(metric_macros("PartFourChosen", results[chosen]["test"], results[chosen]["train"],
                                y_mean=float(y.mean())))
    macros["PartFourChosenExtrap"] = fmt_num(extrap[chosen] / SCALE)

    rows = []
    for name in models:
        rows.append((f"{model_label(name)} (train)", results[name]["train"]))
        rows.append((f"{model_label(name)} (test)", results[name]["test"]))
    table = metrics_table_tex(
        rows,
        "M\\'etriques des mod\\`eles lin\\'eaire et polynomiaux (unit\\'es de \\texttt{Value})",
        "tab:p4_metrics",
    )
    write_tex("part4_metrics.tex", macros, table)
    print("\nPart 4 finished.")


if __name__ == "__main__":
    main()
