# Checklist des captures d'écran

Les captures sont **optionnelles** : `report/main.tex` compile sans elles et les insère
automatiquement dès que le fichier existe dans `report/screenshots/` sous le **nom exact**
indiqué ci-dessous (garde `\IfFileExists`). Format : PNG. Après avoir ajouté des captures,
relance `make report`.

Conseils :
- Agrandis la police du terminal (≥ 14 pt) et recadre sur la zone utile ; les captures sont
  réduites à ~96 % de la largeur de page (hauteur max. 40 % de la page).
- Pour afficher une zone précise d'un log, utilise les commandes `sed -n '/début/,/fin/p'`
  ci-dessous (les bannières des scripts servent de repères). Lance d'abord `make all`.
- Pour ouvrir une figure : `xdg-open <fichier>` (Linux), `open <fichier>` (macOS),
  `explorer.exe <fichier>` (WSL).
- Ne capture jamais tes identifiants Kaggle (variables d'environnement) : cache-les.

| # | Fichier à créer dans `report/screenshots/` | Section du rapport |
|---|---|---|
| 1 | `e_01_versions.png` | 1.1 Environnement |
| 2 | `e_02_tree_and_data.png` | 1.1 Environnement |
| 3 | `p1_00_columns_check.png` | 2.1 (et 1.2) |
| 4 | `p1_01_salary_head_describe.png` | 2.1 |
| 5 | `p1_02_insurance_head_describe.png` | 2.1 |
| 6 | `p1_03_salary_scatter.png` | 2.3 |
| 7 | `p1_04_scatter_matrix.png` | 2.4 |
| 8 | `p2_01_coefficients_predictions.png` | 3.1 |
| 9 | `p2_02_regression_plot.png` | 3.3 |
| 10 | `p2_03_metrics.png` | 3.4 |
| 11 | `p3_01_eda_terminal.png` | 4.1 |
| 12 | `p3_02_eda_overview.png` | 4.1 |
| 13 | `p3_03_corr_heatmap.png` | 4.1 |
| 14 | `p3_04_feature_selection_terminal.png` | 4.2 |
| 15 | `p3_05_feature_scores_plot.png` | 4.2 |
| 16 | `p3_06_scaler_terminal.png` | 4.3 |
| 17 | `p3_07_coefficients_terminal.png` | 4.4 |
| 18 | `p3_08_actual_vs_predicted.png` | 4.4 |
| 19 | `p3_09_metrics.png` | 4.5 |
| 20 | `p4_01_models_terminal.png` | 5.1 |
| 21 | `p4_02_test_predictions.png` | 5.2 |
| 22 | `p4_05_comparison_plot.png` | 5.3 |
| 23 | `p4_03_metrics.png` | 5.4 |
| 24 | `p4_04_degree_selection.png` | 5.4 |
| 25 | `g_01_make_all.png` | Annexe A |
| 26 | `g_02_git_log_graph.png` | Annexe A |
| 27 | `g_03_github_repo.png` | Annexe A |
| 28 | `g_04_pdf_compiled.png` | Annexe A |

---

## Environnement

### 1. `e_01_versions.png`
- **Commande** : `python --version && pip freeze | grep -iE "^(numpy|pandas|scikit-learn|matplotlib)=="`
  (venv activé : `source .venv/bin/activate`)
- **Visible** : version de Python et versions de numpy, pandas, scikit-learn, matplotlib.
- **Rapport** : section 1.1 (complète la phrase `[À COMPLÉTER … versions]`).

### 2. `e_02_tree_and_data.png`
- **Commande** : `find . -maxdepth 2 -not -path './.git*' -not -path './.venv*' -not -path './report/build*' | sort && ls -l data/raw`
- **Visible** : arborescence du dépôt et les trois CSV présents dans `data/raw/` (non versionnés).
- **Rapport** : section 1.1.

## Partie 1

### 3. `p1_00_columns_check.png`
- **Commande** : `make part1 2>&1 | grep "columns found"` (ou capture du début de `make part1`)
- **Visible** : les lignes `[salary] columns found: [...]` et `[insurance] columns found: [...]`, qui prouvent la vérification des noms de colonnes.
- **Rapport** : sections 1.2 et 2.1.

### 4. `p1_01_salary_head_describe.png`
- **Commande** : `sed -n '/Salary - exploration/,/Insurance - exploration/p' results/logs/part1.txt | head -60` puis, dans la même capture ou une suivante, `grep -A12 "salary - describe" results/logs/part1.txt`
- **Visible** : `shape`, `head()`, `info()`, valeurs manquantes, doublons, puis `describe()` du jeu Salary.
- **Rapport** : section 2.1.

### 5. `p1_02_insurance_head_describe.png`
- **Commande** : `sed -n '/Insurance - exploration/,$p' results/logs/part1.txt | head -70`
- **Visible** : `shape`, `head()`, `info()`, manquants, doublons, `describe()` et effectifs des variables catégorielles du jeu Insurance.
- **Rapport** : section 2.1 (la grille de lecture du résumé statistique est en 2.2).

### 6. `p1_03_salary_scatter.png`
- **Fichier à ouvrir** : `report/figures/p1_salary_scatter.png`
- **Visible** : nuage de points Experience/Salary, titre, axes, légende avec le r de Pearson.
- **Rapport** : section 2.3 (la figure elle-même est déjà incluse par le script ; cette capture montre le fichier ouvert).

### 7. `p1_04_scatter_matrix.png`
- **Fichier à ouvrir** : `report/figures/p1_insurance_scatter_matrix.png`
- **Visible** : la scatter_matrix complète (age, bmi, children, charges), points colorés selon `smoker`, légende.
- **Rapport** : section 2.4.

## Partie 2

### 8. `p2_01_coefficients_predictions.png`
- **Commande** : `sed -n '/Item 1 - train/,/Item 3 - regression plot/p' results/logs/part2.txt`
- **Visible** : tailles train/test, équation du modèle (intercept et pente), tableau des prédictions de test.
- **Rapport** : section 3.1.

### 9. `p2_02_regression_plot.png`
- **Fichier à ouvrir** : `report/figures/p2_regression_plot.png`
- **Visible** : droite de régression, points train et test, résidus du test, légende.
- **Rapport** : section 3.3.

### 10. `p2_03_metrics.png`
- **Commande** : `sed -n '/Item 4 - evaluation/,$p' results/logs/part2.txt` et `cat results/part2_metrics.json`
- **Visible** : lignes `TRAIN` et `TEST` avec MSE, RMSE, MAE, R², baseline RMSE, RMSE/moyenne.
- **Rapport** : section 3.4.

## Partie 3

### 11. `p3_01_eda_terminal.png`
- **Commande** : `sed -n '/Item 1 - EDA/,/Item 2 - feature selection/p' results/logs/part3.txt | head -90`
- **Visible** : `info()`, `describe()`, effectifs des catégories, skewness de `charges`, colonnes encodées, corrélations avec `charges`.
- **Rapport** : section 4.1.

### 12. `p3_02_eda_overview.png`
- **Fichier à ouvrir** : `report/figures/p3_eda_overview.png`
- **Visible** : histogramme de `charges`, boxplot selon `smoker`, nuages `bmi` et `age` colorés par `smoker`.
- **Rapport** : section 4.1.

### 13. `p3_03_corr_heatmap.png`
- **Fichier à ouvrir** : `report/figures/p3_corr_heatmap.png`
- **Visible** : matrice de corrélation annotée des variables encodées.
- **Rapport** : section 4.1.

### 14. `p3_04_feature_selection_terminal.png`
- **Commande** : `sed -n '/Item 2 - feature selection/,/Item 3 - standardisation/p' results/logs/part3.txt`
- **Visible** : tableau des scores (|corr|, F, p-value, information mutuelle, rangs), top-3 de chaque critère, ligne `SELECTED FEATURES`.
- **Rapport** : section 4.2 (scores de feature selection).

### 15. `p3_05_feature_scores_plot.png`
- **Fichier à ouvrir** : `report/figures/p3_feature_scores.png`
- **Visible** : trois barplots (corrélation, F-score, information mutuelle), variables retenues en rouge.
- **Rapport** : section 4.2.

### 16. `p3_06_scaler_terminal.png`
- **Commande** : `sed -n '/Item 3 - standardisation/,/Item 4 - LinearRegression/p' results/logs/part3.txt`
- **Visible** : `Scaler mean_` et `Scaler scale_` (ajustés sur le train).
- **Rapport** : section 4.3.

### 17. `p3_07_coefficients_terminal.png`
- **Commande** : `sed -n '/Item 4 - LinearRegression/,/Item 6 - result plot/p' results/logs/part3.txt`
- **Visible** : coefficients standardisés et en unités d'origine, intercepts, prédictions de test, `Scaling invariance check`.
- **Rapport** : section 4.4 (coefficients des modèles).

### 18. `p3_08_actual_vs_predicted.png`
- **Fichier à ouvrir** : `report/figures/p3_actual_vs_predicted.png`
- **Visible** : réel vs prédit avec diagonale, résidus vs prédit, points colorés selon `smoker`.
- **Rapport** : section 4.4.

### 19. `p3_09_metrics.png`
- **Commande** : `sed -n '/Item 7 - evaluation/,$p' results/logs/part3.txt` et `cat results/part3_metrics.json`
- **Visible** : `TRAIN` et `TEST` (MSE, RMSE, MAE, R²), baseline RMSE, RMSE/moyenne.
- **Rapport** : section 4.5.

## Partie 4

### 20. `p4_01_models_terminal.png`
- **Commande** : `sed -n '/Item 1 - train/,/Item 2 - predictions/p' results/logs/part4.txt`
- **Visible** : tailles train/test, coefficients et intercept de chaque modèle (linéaire, d = 2, 3, 4).
- **Rapport** : section 5.1 (coefficients des modèles).

### 21. `p4_02_test_predictions.png`
- **Commande** : `sed -n '/Item 2 - predictions/,/Item 4 - evaluation/p' results/logs/part4.txt`
- **Visible** : tableau des prédictions de test des quatre modèles.
- **Rapport** : section 5.2.

### 22. `p4_05_comparison_plot.png`
- **Fichier à ouvrir** : `report/figures/p4_comparison.png`
- **Visible** : panneau gauche (ajustement, modèle retenu en trait épais), panneau droit (extrapolation, zone grisée).
- **Rapport** : section 5.3 (plot de comparaison de la Partie 4).

### 23. `p4_03_metrics.png`
- **Commande** : `sed -n '/Item 4 - evaluation/,/Degree selection/p' results/logs/part4.txt` et `cat results/part4_metrics.json`
- **Visible** : MSE, RMSE, MAE, R² en `TRAIN` et `TEST` pour les quatre modèles.
- **Rapport** : section 5.4.

### 24. `p4_04_degree_selection.png`
- **Commande** : `sed -n '/Degree selection/,$p' results/logs/part4.txt`
- **Visible** : meilleure RMSE polynomiale, seuil +5 %, degrés éligibles, `CHOSEN DEGREE`, prédictions hors données.
- **Rapport** : section 5.4.

## Contrôles globaux

### 25. `g_01_make_all.png`
- **Commande** : `make all 2>&1 | tail -25; echo "exit code: ${PIPESTATUS[0]}"`
- **Visible** : `Part 4 finished.` et `exit code: 0` (les quatre parties passent).
- **Rapport** : Annexe A.

### 26. `g_02_git_log_graph.png`
- **Commande** : `git log --oneline --graph --decorate`
- **Visible** : l'historique des commits Conventional Commits, dans l'ordre du workflow, avec le tag `v1.0.0`.
- **Rapport** : Annexe A.

### 27. `g_03_github_repo.png`
- **À faire** : ouvrir la page du dépôt sur GitHub (après le push).
- **Visible** : nom du dépôt, description, topics, arborescence, README rendu ; **aucun** CSV dans `data/raw/`.
- **Rapport** : Annexe A.

### 28. `g_04_pdf_compiled.png`
- **Commande** : `make report` puis ouvrir `report/build/main.pdf`
- **Visible** : page de garde et une page de résultats avec de vrais chiffres (sans `[?]`).
- **Rapport** : Annexe A.
