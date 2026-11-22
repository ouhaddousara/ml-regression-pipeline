# data/

Les fichiers de données ne sont **pas** versionnés (voir `.gitignore`).
Place les trois CSV dans `data/raw/` (manuellement ou via `make data`).

| Dataset | Source | Fichier attendu | Colonnes attendues (hypothèses à vérifier) |
|---|---|---|---|
| Experience / Salary | Kaggle `rohankayan/years-of-experience-and-salary-dataset` | `Salary_dataset.csv` ou `Salary_Data.csv` | `YearsExperience`, `Salary` |
| Insurance | Kaggle `sinaasappel/tutorial-multiple-regression` | `insurance.csv` | `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges` |
| China GDP | URL de l'énoncé (IBM Cognitive Class) | `china_gdp.csv` | `Year`, `Value` |

- Les scripts acceptent aussi tout CSV dont le nom contient `salary`, `insurance` ou `gdp`.
- Une colonne d'index parasite (`Unnamed: 0`) est ignorée automatiquement.
- Si une colonne attendue manque, le script affiche `df.columns` et s'arrête avec un message explicite.

## Téléchargement via Kaggle CLI

Les identifiants restent dans l'environnement du terminal, jamais dans un fichier :

    export KAGGLE_USERNAME="..."
    read -rsp "Kaggle key: " KAGGLE_KEY; echo; export KAGGLE_KEY
    make data

## Téléchargement manuel

Télécharge chaque dataset depuis le navigateur, dézippe, puis place le CSV dans `data/raw/`.
