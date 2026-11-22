"""Part 3 - Multiple linear regression (Insurance: predict `charges`).

Assignment items covered
  1. EDA + conclusions (conclusions are written in the report)
  2. select 3 features by importance (>= 2 methods) and justify
  3. standardise / normalise the chosen features and justify
  4. train LinearRegression      5. predict the test set
  6. plot the result             7. MSE / RMSE / MAE + interpretation

Run from the repository root:  python src/part3_multiple_regression.py
"""
from __future__ import annotations

from functools import partial

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
    tex_escape,
    write_tex,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET = "charges"
CATEGORICAL = ["sex", "smoker", "region"]
CONTINUOUS = ["age", "bmi"]  # everything else is treated as discrete by mutual information
N_SELECT = 3
PALETTE = ["tab:blue", "tab:red", "tab:green", "tab:orange"]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_eda_overview(df: pd.DataFrame, skew: float) -> None:
    smoker = df["smoker"].astype(str)
    cats = sorted(smoker.unique())
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))

    axes[0, 0].hist(df[TARGET], bins=40, color="tab:blue", edgecolor="white",
                    label=f"charges (skewness = {skew:.2f})")
    axes[0, 0].set_title("Distribution de la cible (charges)")
    axes[0, 0].set_xlabel("charges")
    axes[0, 0].set_ylabel("Effectif")
    axes[0, 0].legend()

    bp = axes[0, 1].boxplot([df.loc[smoker == c, TARGET] for c in cats], patch_artist=True)
    for patch, col in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    axes[0, 1].set_xticks(range(1, len(cats) + 1))
    axes[0, 1].set_xticklabels(cats)
    axes[0, 1].set_title("charges selon smoker")
    axes[0, 1].set_xlabel("smoker")
    axes[0, 1].set_ylabel("charges")
    axes[0, 1].legend([Patch(facecolor=c, alpha=0.6) for c in PALETTE[: len(cats)]],
                      [f"smoker = {c}" for c in cats])

    for ax, feature in ((axes[1, 0], "bmi"), (axes[1, 1], "age")):
        for cat, col in zip(cats, PALETTE):
            mask = smoker == cat
            ax.scatter(df.loc[mask, feature], df.loc[mask, TARGET], s=14, alpha=0.5,
                       color=col, label=f"smoker = {cat}")
        ax.set_title(f"charges en fonction de {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("charges")
        ax.legend()

    fig.suptitle("Insurance - vue d'ensemble de l'EDA", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, "p3_eda_overview.png")


def plot_corr_heatmap(corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Corrélation de Pearson")
    ax.set_title("Matrice de corrélation (variables encodées)")
    ax.set_xlabel("Variables")
    ax.set_ylabel("Variables")
    save_fig(fig, "p3_corr_heatmap.png")


def plot_feature_scores(scores: pd.DataFrame, selected: list[str]) -> None:
    metrics = [
        ("abs_corr", "|Corrélation de Pearson| avec charges"),
        ("f_score", "F-score (f_regression)"),
        ("mutual_info", "Information mutuelle"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (col, title) in zip(axes, metrics):
        s = scores[col].sort_values()
        colors = ["tab:red" if f in selected else "tab:gray" for f in s.index]
        ax.barh(s.index, s.values, color=colors)
        ax.set_title(title)
        ax.set_xlabel("Score")
        ax.set_ylabel("Variable")
    fig.suptitle("Scores de sélection de variables (calculés sur le jeu d'entraînement)", fontsize=13)
    fig.legend(
        handles=[Patch(color="tab:red", label="Variable retenue"),
                 Patch(color="tab:gray", label="Non retenue")],
        loc="lower center", ncol=2,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    save_fig(fig, "p3_feature_scores.png")


def plot_predictions(y_test: pd.Series, pred_test: np.ndarray, smoker_test: pd.Series) -> None:
    cats = sorted(smoker_test.unique())
    residuals = y_test.values - pred_test
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for cat, col in zip(cats, PALETTE):
        mask = (smoker_test == cat).values
        axes[0].scatter(y_test.values[mask], pred_test[mask], s=22, alpha=0.65,
                        color=col, label=f"smoker = {cat}")
        axes[1].scatter(pred_test[mask], residuals[mask], s=22, alpha=0.65,
                        color=col, label=f"smoker = {cat}")
    lims = [min(y_test.min(), pred_test.min()), max(y_test.max(), pred_test.max())]
    axes[0].plot(lims, lims, "k--", label="Prédiction parfaite (y = x)")
    axes[0].set_title("Valeurs réelles vs prédites (jeu de test)")
    axes[0].set_xlabel("charges réelles")
    axes[0].set_ylabel("charges prédites")
    axes[0].legend()
    axes[1].axhline(0, color="k", linestyle="--", label="Erreur nulle")
    axes[1].set_title("Résidus vs valeurs prédites (jeu de test)")
    axes[1].set_xlabel("charges prédites")
    axes[1].set_ylabel("Résidu (réel - prédit)")
    axes[1].legend()
    fig.suptitle("Régression linéaire multiple - Insurance", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "p3_actual_vs_predicted.png")


# --------------------------------------------------------------------------- #
# Main workflow
# --------------------------------------------------------------------------- #
def main() -> None:
    ensure_dirs()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    # ------------------------------------------------------------- 1. EDA
    section("Item 1 - EDA")
    df = load_dataset("insurance")
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"Rows after dropping missing values: {len(df)} / {n_before}")
    print("\n-- info()")
    df.info()
    print("\n-- describe()")
    print(df.describe().T)
    print(f"\nduplicated rows: {int(df.duplicated().sum())}")
    for col in CATEGORICAL:
        print(f"\n{col}:\n{df[col].value_counts()}")
    skew = float(df[TARGET].skew())
    print(f"\nSkewness of {TARGET}: {skew:.4f}")

    # Encoding: one-hot with drop_first (avoids the dummy-variable trap).
    encoded = pd.get_dummies(df, columns=CATEGORICAL, drop_first=True, dtype=int)
    print("\nEncoded columns:", list(encoded.columns))
    corr = encoded.corr()
    print(f"\nCorrelation with {TARGET} (full data, EDA only):")
    print(corr[TARGET].drop(TARGET).sort_values(ascending=False))
    plot_eda_overview(df, skew)
    plot_corr_heatmap(corr)

    # -------------------------------------------- split BEFORE any fitting step
    X = encoded.drop(columns=[TARGET])
    y = encoded[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\ntrain = {len(X_train)} rows, test = {len(X_test)} rows "
          f"(test_size={TEST_SIZE}, random_state={RANDOM_STATE})")

    # -------------------------------------------------- 2. feature selection
    # All scores are computed on the TRAINING set only, so the test set plays no
    # role in the choice of the features (no selection leakage).
    section("Item 2 - feature selection (3 criteria, TRAIN set only)")
    abs_corr = X_train.corrwith(y_train).abs().reindex(X_train.columns)
    f_sel = SelectKBest(score_func=f_regression, k="all").fit(X_train, y_train)
    # Mutual information: dummies and `children` are flagged as discrete features.
    discrete_mask = np.array([c not in CONTINUOUS for c in X_train.columns])
    mi_func = partial(mutual_info_regression, discrete_features=discrete_mask,
                      random_state=RANDOM_STATE)
    mi_sel = SelectKBest(score_func=mi_func, k="all").fit(X_train, y_train)

    scores = pd.DataFrame(
        {
            "abs_corr": abs_corr.values,
            "f_score": f_sel.scores_,
            "f_pvalue": f_sel.pvalues_,
            "mutual_info": mi_sel.scores_,
        },
        index=X_train.columns,
    )
    criteria = ["abs_corr", "f_score", "mutual_info"]
    for c in criteria:
        scores[f"rank_{c}"] = scores[c].rank(ascending=False, method="min")
    scores["mean_rank"] = scores[[f"rank_{c}" for c in criteria]].mean(axis=1)
    # Decision rule (explicit): keep the N_SELECT features with the best (lowest) mean
    # rank over the three criteria; ties are broken by the larger |correlation|.
    scores = scores.sort_values(["mean_rank", "abs_corr"], ascending=[True, False])
    selected = list(scores.index[:N_SELECT])
    print(scores.to_string(float_format=lambda v: f"{v:.4g}"))
    for c in criteria:
        print(f"Top-{N_SELECT} by {c}: {list(scores[c].nlargest(N_SELECT).index)}")
    print(f"\nSELECTED FEATURES (lowest mean rank): {selected}")
    plot_feature_scores(scores, selected)

    sel_table = scores.copy()
    sel_table["Retenue"] = np.where(sel_table.index.isin(selected), "oui", "non")
    sel_table = sel_table[["abs_corr", "f_score", "f_pvalue", "mutual_info",
                           "rank_abs_corr", "rank_f_score", "rank_mutual_info",
                           "mean_rank", "Retenue"]]
    sel_table.columns = ["$|r|$", "$F$", "$p$-value", "MI", "rang $|r|$", "rang $F$",
                         "rang MI", "rang moyen", "Retenue"]
    sel_tex = df_to_latex(
        sel_table,
        "Scores de s\\'election de variables (jeu d'entra\\^inement)",
        "tab:p3_feature_selection",
        index=True,
        index_header="Variable",
        resize=True,
    )
    write_tex("part3_feature_selection.tex", {}, sel_tex)
    save_json("part3_feature_selection.json",
              {"selected": selected, "scores": scores.to_dict(orient="index")})

    # ------------------------------------------------------- 3. standardisation
    # Why StandardScaler (z-score) rather than MinMaxScaler:
    #  * the selected features live on very different scales (e.g. age in years,
    #    bmi, a 0/1 dummy), and standardising puts every coefficient in "effect of
    #    +1 standard deviation" units, which makes them comparable;
    #  * min-max scaling is driven by the extreme values (bmi has outliers), whereas
    #    z-scores do not assume bounded ranges;
    #  * centring makes the intercept the prediction for the average profile.
    # Honest note: OLS predictions are invariant to affine rescaling, so the test
    # metrics must not change; we verify this below (scaling_invariance_max_abs_diff).
    # The scaler is fitted on the TRAINING set only and then applied to the test set,
    # so no test statistic leaks into training (no data leakage).
    section("Item 3 - standardisation (StandardScaler fitted on TRAIN only)")
    X_train_sel, X_test_sel = X_train[selected], X_test[selected]
    scaler = StandardScaler().fit(X_train_sel)
    X_train_s = pd.DataFrame(scaler.transform(X_train_sel), columns=selected,
                             index=X_train_sel.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test_sel), columns=selected,
                            index=X_test_sel.index)
    print("Scaler mean_ (train):", dict(zip(selected, np.round(scaler.mean_, 4))))
    print("Scaler scale_ (train):", dict(zip(selected, np.round(scaler.scale_, 4))))

    # ---------------------------------------------------------- 4. training
    section("Item 4 - LinearRegression fit")
    model = LinearRegression().fit(X_train_s, y_train)
    coef_std = pd.Series(model.coef_, index=selected)
    coef_orig = coef_std / scaler.scale_
    intercept_orig = float(model.intercept_ - np.sum(coef_std.values * scaler.mean_ / scaler.scale_))
    coef_df = pd.DataFrame(
        {
            "Coef. standardisé": coef_std.values,
            "Coef. unité d'origine": coef_orig.values,
            "Moyenne (train)": scaler.mean_,
            "Écart-type (train)": scaler.scale_,
        },
        index=selected,
    )
    print(coef_df.to_string(float_format=lambda v: f"{v:.6g}"))
    print(f"Intercept (standardised space): {model.intercept_:.6g}")
    print(f"Intercept (original units):     {intercept_orig:.6g}")
    write_tex(
        "part3_coefficients.tex",
        {},
        df_to_latex(coef_df, "Coefficients du mod\\`ele de r\\'egression multiple",
                    "tab:p3_coefficients", index=True, index_header="Variable", resize=True),
    )

    # --------------------------------------------------------- 5. predictions
    section("Item 5 - predictions on the test set")
    pred_train = model.predict(X_train_s)
    pred_test = model.predict(X_test_s)
    pred_df = pd.DataFrame(
        {"Réel": y_test.values, "Prédit": pred_test, "Erreur (réel - prédit)": y_test.values - pred_test},
        index=y_test.index,
    )
    print(pred_df.head(10).to_string(float_format=lambda v: f"{v:.2f}"))
    pred_df.to_csv(RES_DIR / "part3_predictions.csv", index_label="row_index")
    write_tex(
        "part3_predictions.tex",
        {},
        df_to_latex(pred_df.head(10), "Dix premi\\`eres pr\\'edictions du jeu de test (Partie 3)",
                    "tab:p3_predictions", index=False),
    )

    raw_model = LinearRegression().fit(X_train_sel, y_train)
    max_diff = float(np.max(np.abs(raw_model.predict(X_test_sel) - pred_test)))
    print(f"\nScaling invariance check: max |pred(scaled) - pred(unscaled)| = {max_diff:.3e}")

    # ------------------------------------------------------------ 6. figure
    section("Item 6 - result plot")
    smoker_test = df.loc[X_test.index, "smoker"].astype(str)
    plot_predictions(y_test, pred_test, smoker_test)

    # ---------------------------------------------------------- 7. evaluation
    section("Item 7 - evaluation (MSE, RMSE, MAE)")
    m_train = regression_metrics(y_train, pred_train)
    m_test = regression_metrics(y_test, pred_test)
    base = baseline_rmse(y_train, y_test)
    print_metrics("TRAIN", m_train)
    print_metrics("TEST", m_test)
    print(f"Baseline RMSE (always predict the train mean): {base:.6g}")
    print(f"Target mean = {y.mean():.6g}, target std = {y.std():.6g}")
    print(f"RMSE(test) / mean(target) = {100 * m_test['rmse'] / y.mean():.2f} %")

    save_json(
        "part3_metrics.json",
        {
            "dataset": "Insurance",
            "target": TARGET,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "encoded_features": list(X.columns),
            "selected_features": selected,
            "selection_rule": "lowest mean rank over |corr|, f_regression, mutual_info (train set)",
            "scaler": {"type": "StandardScaler", "fitted_on": "train only",
                       "mean": dict(zip(selected, scaler.mean_)),
                       "scale": dict(zip(selected, scaler.scale_))},
            "coef_standardised": dict(zip(selected, coef_std.values)),
            "coef_original_units": dict(zip(selected, coef_orig.values)),
            "intercept_standardised": float(model.intercept_),
            "intercept_original_units": intercept_orig,
            "scaling_invariance_max_abs_diff": max_diff,
            "target_skewness": skew,
            "train": m_train,
            "test": m_test,
            "baseline_rmse": base,
            "target_mean": float(y.mean()),
            "target_std": float(y.std()),
        },
    )

    macros = {
        "PartThreeFeatureOne": "\\texttt{" + tex_escape(selected[0]) + "}",
        "PartThreeFeatureTwo": "\\texttt{" + tex_escape(selected[1]) + "}",
        "PartThreeFeatureThree": "\\texttt{" + tex_escape(selected[2]) + "}",
        "PartThreeNFeatures": fmt_num(X.shape[1]),
        "PartThreeSkew": f"{skew:.2f}",
        "PartThreeIntercept": fmt_num(float(model.intercept_)),
        "PartThreeInterceptOrig": fmt_num(intercept_orig),
        "PartThreeMaxScalingDiff": fmt_num(max_diff),
    }
    macros.update(metric_macros("PartThree", m_test, m_train, y_mean=float(y.mean())))
    macros.update(scale_macros("PartThree", y, y_train, y_test))
    table = metrics_table_tex(
        [("Train", m_train), ("Test", m_test)],
        "M\\'etriques de la r\\'egression lin\\'eaire multiple (unit\\'es de \\texttt{charges})",
        "tab:p3_metrics",
    )
    write_tex("part3_metrics.tex", macros, table)
    print("\nPart 3 finished.")


if __name__ == "__main__":
    main()
