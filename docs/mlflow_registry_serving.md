# MLflow Registry Serving

## Pourquoi charger depuis MLflow Model Registry

Le Model Registry permet de decoupler le serving du chemin local d un checkpoint unique. Cela facilite :

- la promotion d une version de modele en Staging ou Production ;
- la tracabilite entre entrainement, registry et serving ;
- la future integration CD / Docker ;
- la reduction du risque d erreur manuelle sur les chemins de checkpoints.

## Strategie retenue

Le serving FastAPI supporte deux modes :

1. `mlflow_registry`
   - charge le modele depuis MLflow via un alias, un stage ou une URI explicite ;
2. `local_checkpoint`
   - charge le checkpoint local historique ;
   - sert aussi de fallback si le Registry est indisponible.

Le fallback local est conserve pour ne pas casser le fonctionnement existant.

## Configuration

Le fichier `configs/serving.yaml` pilote le chargement :

```yaml
model_source: mlflow_registry
mlflow_tracking_uri: ./mlruns
registered_model_name: flood_damage_siamese_resnet18
model_stage: null
model_alias: champion
direct_model_uri: null
fallback_checkpoint_path: outputs/focal_run_b_regularized/checkpoints/best_siamese_resnet18.pt
device: auto
num_classes: 4
```

## Comment FastAPI recupere le modele

- au demarrage, l application tente un prechargement du modele ;
- si `model_source=mlflow_registry`, le loader essaie d abord MLflow ;
- si le chargement MLflow echoue, le loader journalise un warning et bascule sur le checkpoint local ;
- les endpoints `/health` et `/model-info` exposent la source reellement utilisee.

## Endpoint utile

`GET /model-info` retourne notamment :

- `model_source_requested`
- `model_source_used`
- `registered_model_name`
- `model_stage`
- `model_alias`
- `model_uri`
- `checkpoint_path_used`
- `device`
- `loaded`
- `load_error`
- `fallback_warning`

## Enregistrement / promotion du modele

Le projet contient deja `src/mlops/register_model.py` pour enregistrer le modele final Run B au format `mlflow.pytorch.log_model(...)`.

Tant que le modele est bien enregistre dans MLflow sous forme PyTorch complete, le serving peut le charger directement via `mlflow.pytorch.load_model(...)`.

Si le Registry ne contient encore qu un checkpoint non emballe, il faut soit :

- enregistrer proprement le modele PyTorch complet avec `register_model.py` ;
- soit continuer temporairement en fallback checkpoint local.

## Commandes de test

### Import API

```bash
python -c "import src.serving.api; print('api import ok')"
```

### Lancer FastAPI

```bash
uvicorn src.serving.app:app --host 127.0.0.1 --port 8000
```

### Tester l etat du modele

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model-info
```

## Impact futur sur le CD Docker

Cette adaptation prepare une etape de CD plus propre :

- le conteneur n a plus besoin de dependre uniquement d un fichier `.pt` fixe ;
- il peut consommer un modele promu dans MLflow Registry ;
- le fallback local reste disponible pour les environnements hors registry.
