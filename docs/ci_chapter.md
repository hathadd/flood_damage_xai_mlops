# 6. Continuous Integration

## 6.1 Introduction

Ce chapitre présente la stratégie d’intégration continue mise en place pour le projet `flood_damage_xai_mlops`. L’objectif est de documenter la structure réelle du pipeline GitHub Actions, sa place dans l’architecture MLOps, ainsi que les validations automatiques qui garantissent l’intégrité du code et la disponibilité des artefacts de service.

## 6.2 CI dans les systèmes MLOps

Dans un projet MLOps, la CI permet de vérifier que les évolutions de code ne cassent pas :
- la compilation Python,
- les importations critiques,
- les points d’entrée des services,
- les fichiers de configuration et de déploiement.

Le pipeline CI du projet n’est pas un mécanisme de validation des performances du modèle, mais un niveau de confiance essentiel avant d’activer des workflows plus coûteux comme l’entraînement ou le déploiement.

## 6.3 Besoin d'une CI dans `flood_damage_xai_mlops`

Pour `flood_damage_xai_mlops`, la CI est nécessaire car le projet combine plusieurs briques :
- une API FastAPI (`src/serving/app.py`, `src/serving/api.py`),
- une interface Streamlit (`src/demo/streamlit_app.py`),
- des scripts de monitoring et de données,
- des artefacts Docker (`Dockerfile.api`, `Dockerfile.streamlit`, `docker-compose.yml`).

Une CI permet de s’assurer que ces composants restent cohérents à chaque fusion sur `main`.

## 6.4 Architecture GitHub Actions

Le workflow GitHub Actions est défini dans `.github/workflows/ci.yml`.

- Déclencheurs : `push` et `pull_request` sur la branche `main`
- Runner : `ubuntu-latest`
- Environnement Python : `3.11`
- Variables : `PYTHONPATH` positionné sur le workspace GitHub

### Figure 1

![Figure 1 - Architecture du workflow CI](docs/images/ci_workflow.svg)

> Légende : diagramme de la séquence d’exécution du workflow GitHub Actions.

## 6.5 Description détaillée du workflow

Le workflow comprend les étapes suivantes :

1. `Checkout repository`
   - Récupération du dépôt via `actions/checkout@v4`.
2. `Set up Python 3.11`
   - Installation de Python 3.11 avec `actions/setup-python@v5`.
3. `Upgrade pip`
   - Mise à jour de l’outil d’installation Python.
4. `Install dependencies`
   - Installation de `requirements-ci.txt`.
5. `Verify essential project files`
   - Vérification de l’existence des fichiers critiques.
6. `Compile Python sources`
   - Compilation de `src`, `tests` et `scripts` avec `python -m compileall`.
7. `Run lightweight CI tests`
   - Exécution de `pytest -q tests/test_imports.py tests/test_ci_readiness.py`.

### Figure 2

![Figure 2 - Étapes détaillées du workflow CI](docs/images/ci_steps.svg)

> Légende : vue détaillée des étapes du workflow, de l’installation des dépendances à l’exécution des tests.

## 6.6 Validation automatisée du projet

La CI automatise plusieurs validations clés :

- présence des fichiers de configuration : `configs/data.yaml`, `configs/serving.yaml`, `configs/monitoring.yaml`;
- présence des fichiers Docker et de documentation (`Dockerfile.api`, `Dockerfile.streamlit`, `docker-compose.yml`, `docs/ci_docker.md`);
- compilation du code Python pour détecter les erreurs de syntaxe en amont;
- import des modules critiques du projet.

La validation structurale protège le projet contre les ruptures de packaging et la détérioration de l’architecture logicielle.

## 6.7 Tests implémentés

Deux suites de tests légers sont exécutées :

### 6.7.1 `tests/test_imports.py`

Ce test vérifie :
- l’importabilité de modules essentiels tels que `src.data.dataset`, `src.training.losses`, `src.serving.app`, `src.demo.streamlit_app`, `src.monitoring.utils`, et d’autres modules de monitoring;
- l’existence des fichiers essentiels listés dans `ESSENTIAL_FILES`.

### 6.7.2 `tests/test_ci_readiness.py`

Ce test vérifie :
- que l’application FastAPI est bien importable et expose les routes critiques suivantes :
  - `/`
  - `/health`
  - `/model-info`
  - `/predict`
  - `/predict-scene`
  - `/explain-building`
- que les scripts de monitoring exportent des points d’entrée `main` et `parse_args` pour :
  - `build_reference_dataset`
  - `collect_inference_logs`
  - `evidently_data_quality`
  - `evidently_data_drift`
- l’existence des artefacts Docker/CI additionnels.

### Figure 3

![Figure 3 - Couverture des tests CI](docs/images/ci_test_coverage.svg)

> Légende : cartes de la couverture fonctionnelle des tests legers exécutés en CI.

## 6.8 Retour d'expérience et résolution des erreurs

Pendant l’implémentation, les erreurs typiques identifiées par la CI sont :

- fichiers de configuration manquants ou déplacés,
- erreurs de syntaxe dans les modules Python,
- importations invalides liées à des changements de structure de package,
- absence de routes API attendues par le service.

Ces problèmes sont détectés tôt, ce qui évite des ruptures plus coûteuses lors du déploiement ou du packaging Docker.

## 6.9 Résultats obtenus

La CI actuelle permet de garantir :

- l’intégrité du repository à chaque modification de `main`,
- la compatibilité minimale des modules critiques,
- la disponibilité des artefacts de déploiement Docker et de la documentation associée,
- une base stable pour des ajouts futurs tels que des tests d’intégration ou des validations ML plus lourdes.

## 6.10 Analyse critique

Cette CI présente des points forts :
- légèreté,
- rapidité d’exécution,
- couverture utile de la structure du projet.

Mais elle a aussi des limites notables :
- elle ne mesure pas la qualité des modèles ni les performances prédictives,
- elle ne teste pas les flux d’inférence en bout en bout,
- elle ne couvre pas la gestion des données ni les jeux d’essais réels.

## 6.11 Limites

Les principales limites de l’intégration continue actuelle sont :

- absence de validation des pipelines d’entraînement,
- absence de tests d’intégration API complets,
- dépendance à une simple compilation et à des importations statiques,
- pas de vérifications automatiques des données de production ou des métriques de dérive.

## 6.12 Conclusion

Pour le projet `flood_damage_xai_mlops`, la CI implémentée apporte une base solide pour la qualité logicielle. Elle est adaptée à un premier niveau de maturité MLOps et prépare le terrain pour des extensions futures : tests d’intégration, validation de modèles, monitoring des données, et déploiement end-to-end.

> Recommandation : étendre cette CI avec des tests fonctionnels FastAPI et des vérifications de pipeline ML pour atteindre un niveau de maturité supérieur.
