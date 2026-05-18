# Guide de rédaction PFE — Partie entraînement et expérimentation

## Objectif du document
Ce document sert de **guide chronologique, factuel et cohérent** pour rédiger la partie **entraînement, expérimentation et évaluation** du projet `flood_damage_xai_mlops`.

Il est conçu pour être utilisé de deux manières :

- comme **plan de rédaction** pour écrire le chapitre manuellement ;
- comme **contexte source fiable** à fournir à Claude/Cowork afin de générer une étude académique solide **sans invention**.

Le but est de produire un chapitre PFE centré sur :

- le modèle **Siamese ResNet18** ;
- la logique des **runs A, B et C** ;
- la pipeline de données et d’augmentation ;
- le cadre **MLOps** avec **MLflow Tracking**, artefacts et sélection du modèle final ;
- l’analyse comparative menant au choix final de **Run B**.

---

## 1. Règles de rédaction à imposer à Claude/Cowork

### 1.1. Règles absolues
Le texte généré doit respecter les règles suivantes :

- Ne **jamais inventer** de résultats, d’hyperparamètres, de run_id ou de métriques absentes.
- Ne **jamais confondre** les configurations des runs.
- Utiliser uniquement les informations vérifiées à partir des artefacts du projet.
- Si une information n’est pas entièrement confirmée localement, la marquer explicitement comme :
  - `[à compléter]`
  - ou `non vérifié localement`
- Distinguer clairement :
  - les **métriques de validation**
  - les **métriques de test**
  - les **hyperparamètres d’entraînement**
  - les **artefacts MLOps**

### 1.2. Style attendu
Le style doit être :

- académique
- rigoureux
- explicatif
- chronologique
- cohérent avec un mémoire de PFE en IA / Computer Vision / MLOps

Le texte doit :

- introduire les concepts avant les résultats ;
- expliquer la motivation de chaque choix ;
- interpréter les métriques et pas seulement les citer ;
- conclure chaque sous-partie par une lecture critique.

---

## 2. Contexte général du chapitre

Le chapitre à produire doit s’intituler dans l’esprit de :

**Expérimentation, Entraînement et Évaluation**

Sous-titre possible :

**Étude comparative de trois architectures pour la classification multi-classes des dommages aux bâtiments à partir d’images satellitaires bi-temporelles dans un cadre MLOps**

Contexte scientifique :

- tâche : classification de dommages à l’échelle bâtiment ;
- données : paires d’images satellitaires **pré-catastrophe / post-catastrophe** ;
- dataset utilisé : **xBD flooding subset** ;
- nombre total d’échantillons : **8 471 bâtiments** ;
- classes :
  - `no-damage`
  - `minor-damage`
  - `major-damage`
  - `destroyed`

Contexte technique :

- pipeline PyTorch complète ;
- tracking des entraînements avec **MLflow** ;
- sauvegarde des checkpoints, historiques CSV, courbes et matrices de confusion ;
- évaluation finale sur un **même split test** ;
- modèle final ensuite relié à :
  - XAI Grad-CAM
  - Model Registry
  - serving FastAPI
  - démo Streamlit

---

## 3. Ordre chronologique recommandé du chapitre

L’ordre de rédaction recommandé est le suivant :

1. Introduction du chapitre
2. Cadre MLOps de l’expérimentation
3. Présentation du modèle de base : ResNet18 puis Siamese ResNet18
4. Pipeline de données
5. Prétraitements et data augmentation
6. Configuration d’entraînement commune
7. Run A — baseline
8. Run B — modèle régularisé
9. Run C — BIT Transformer
10. Comparaison globale des trois runs
11. Sélection du modèle final
12. Transition vers XAI, Registry et déploiement

Cet ordre est important car il suit la logique réelle du projet :

- comprendre le backbone ;
- comprendre la forme siamoise ;
- comprendre la donnée ;
- comprendre l’entraînement ;
- analyser les résultats ;
- justifier le choix final.

---

## 4. Partie modèle — ce qu’il faut dire sur ResNet18

### 4.1. Présenter ResNet18 avant le modèle siamois
Avant de parler du modèle final, il faut expliquer ce qu’est **ResNet18**.

Éléments à inclure :

- ResNet18 est un réseau convolutif résiduel introduit pour faciliter l’entraînement de réseaux profonds.
- Son idée centrale est le **residual learning** :
  - au lieu d’apprendre directement `H(x)`, le bloc apprend une fonction résiduelle `F(x)`,
  - et la sortie devient `y = F(x) + x`.
- Les **skip connections** améliorent :
  - la propagation du gradient,
  - la stabilité de l’optimisation,
  - la capacité à entraîner des réseaux plus profonds.

### 4.2. Décrire l’architecture backbone réellement utilisée
Backbone réel dans le projet :

- `Conv1`: `7x7`, 64 filtres, stride 2
- `MaxPool`
- `Layer1`: 2 BasicBlocks
- `Layer2`: 2 BasicBlocks
- `Layer3`: 2 BasicBlocks
- `Layer4`: 2 BasicBlocks
- `Global Average Pooling`
- sortie finale backbone : **512 dimensions**

Point important :

- dans le projet, la couche finale de classification originale de ResNet18 est retirée ;
- le backbone est utilisé comme **extracteur de caractéristiques**.

### 4.3. Dimensions réelles à citer
Pour un crop d’entrée `3 x 224 x 224` :

- sortie backbone = `512-D`
- paramètres backbone = **11,176,512**

Ne pas dire que le backbone est entraîné from scratch pour Run A/B.

Pour **Run A** et **Run B**, les checkpoints sauvegardés indiquent :

- `pretrained=True`

---

## 5. Partie modèle — ce qu’il faut dire sur le Siamese ResNet18

### 5.1. Principe du modèle siamois
Expliquer que le modèle final n’est pas un ResNet18 simple, mais un **Siamese ResNet18**.

Deux entrées :

- image `PRE`
- image `POST`

Les deux branches :

- utilisent le **même backbone ResNet18**
- partagent exactement les **mêmes poids**

Pourquoi ce choix est pertinent :

- les deux images représentent le même bâtiment à deux dates différentes ;
- le partage des poids impose un espace latent commun ;
- cela rend la comparaison temporelle plus cohérente ;
- cela limite aussi le nombre effectif de paramètres indépendants.

### 5.2. Fusion temporelle réelle utilisée dans le projet
Le modèle extrait :

- `pre_features` de dimension `512`
- `post_features` de dimension `512`

Puis il construit :

- `abs(post_features - pre_features)` de dimension `512`

Fusion finale :

- concaténation de :
  - `pre_features`
  - `post_features`
  - `abs(post_features - pre_features)`

Donc :

- vecteur fusionné final = **1536 dimensions**

### 5.3. Tête de classification réelle
La tête de classification réelle est :

- `Linear(1536 -> 512)`
- `ReLU`
- `Dropout(p=dropout)`
- `Linear(512 -> 4)`

Pour **Run B** :

- `dropout = 0.4`

### 5.4. Paramètres réels à citer
Paramètres du modèle Siamese ResNet18 :

- backbone : **11,176,512**
- classifier head : **788,996**
- total trainable params : **11,965,508**

---

## 6. Pipeline de données — ce qu’il faut expliquer

### 6.1. Source de données
Le dataset central pour l’entraînement est :

- `data/splits/metadata_splits.csv`

Chaque ligne correspond à :

- un bâtiment
- un `sample_id`
- un `building_uid`
- un chemin image PRE
- un chemin image POST
- une classe de dommage
- un polygone WKT
- un split : `train`, `val`, `test`

### 6.2. Répartition réelle des splits
Ta rédaction doit citer les valeurs exactes :

- train = **6313**
- val = **1327**
- test = **831**

### 6.3. Distribution réelle des classes
Total :

- `no-damage = 8128`
- `minor-damage = 149`
- `major-damage = 119`
- `destroyed = 75`

Split train :

- `no-damage = 6090`
- `minor-damage = 96`
- `major-damage = 80`
- `destroyed = 47`

Split val :

- `no-damage = 1261`
- `minor-damage = 28`
- `major-damage = 18`
- `destroyed = 20`

Split test :

- `no-damage = 777`
- `minor-damage = 25`
- `major-damage = 21`
- `destroyed = 8`

### 6.4. Message analytique à imposer
Claude doit expliquer clairement que :

- le jeu est **extrêmement déséquilibré** ;
- `no-damage` domine massivement ;
- ce déséquilibre influence :
  - la fonction de perte,
  - le sampler,
  - l’interprétation des métriques.

---

## 7. Data augmentation et prétraitement — ce qu’il faut dire

### 7.1. Idée clé
Il faut distinguer :

- les transformations **géométriques synchronisées**
- les transformations **photométriques indépendantes**

Pourquoi :

- la géométrie doit rester alignée entre PRE et POST ;
- sinon la comparaison temporelle devient incohérente.

### 7.2. Transformations réelles Run A / Run B
Train :

- `Resize(224, 224)`
- `HorizontalFlip(p=0.5)`
- `VerticalFlip(p=0.2)`
- `RandomRotate90(p=0.3)`
- `ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=15, p=0.3)`
- `RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3)`
- `GaussNoise(p=0.2)`
- `Normalize(ImageNet mean/std)`
- `ToTensorV2()`

Val / Test :

- `Resize(224, 224)`
- `Normalize(ImageNet mean/std)`
- `ToTensorV2()`

### 7.3. Ce qu’il faut dire sur le rôle de l’augmentation
Important : ne pas écrire que la data augmentation “corrige” directement le déséquilibre.

Formulation correcte :

- l’augmentation **augmente la diversité apparente des données d’entraînement** ;
- elle **réduit le surapprentissage** ;
- elle **améliore la robustesse** ;
- mais la gestion directe du déséquilibre est surtout assurée par :
  - la **Focal Loss**
  - le **WeightedRandomSampler**
  - les **class weights**

---

## 8. Configuration d’entraînement commune — ce qu’il faut décrire

### 8.1. Éléments généraux
Le chapitre doit expliquer le pipeline d’entraînement standard :

- forward pass
- calcul de la loss
- backward pass
- update optimiseur
- scheduler LR
- mixed precision
- early stopping
- sauvegarde checkpoints
- tracking MLflow

### 8.2. Composants réels du projet
À citer :

- Optimiseur : **AdamW**
- Scheduler principal : **cosine annealing**
- Warmup : **Linear warmup**
- Gradient clipping : `max_norm = 1.0`
- Mixed precision : activée quand GPU CUDA disponible
- Early stopping : basé sur `val_macro_f1`

### 8.3. Gestion du déséquilibre
À expliquer explicitement :

- `WeightedRandomSampler`
- `class_weights`
- `FocalLoss`

Message à faire passer :

- le problème principal du dataset n’est pas seulement la taille, mais aussi le **déséquilibre extrême des classes**

---

## 9. Partie MLOps — ce qu’il faut absolument intégrer

### 9.1. Cadre MLOps réel
Le projet ne s’arrête pas à l’entraînement.
Il faut montrer qu’il s’inscrit dans un cadre **MLOps** avec :

- **MLflow Tracking**
- sauvegarde des paramètres
- sauvegarde des métriques
- sauvegarde des artefacts
- conservation des checkpoints
- évaluation finale versionnée
- futur enregistrement dans le **Model Registry**

### 9.2. Ce que MLflow enregistre dans le projet
Le texte doit expliquer que chaque run journalise :

- paramètres
  - learning rate
  - weight decay
  - batch size
  - dropout
  - epochs
  - gamma focal
  - scheduler
  - warmup
  - early stopping
  - etc.
- métriques par époque
  - `train_loss`
  - `train_accuracy`
  - `train_macro_f1`
  - `val_loss`
  - `val_accuracy`
  - `val_macro_f1`
  - `learning_rate`
- artefacts
  - checkpoints
  - history CSV
  - courbes
  - matrices de confusion

### 9.3. Point important de vérité
Il faut être prudent :

- le dossier `mlruns/` local ne contient **pas forcément tous les run_id finaux vérifiables** pour Run B et Run C
- si Claude doit citer les `run_id`, il faut lui dire :
  - utiliser `[à compléter depuis l’interface MLflow]`
  - sauf si l’identifiant a été vérifié explicitement

Donc il faut **interdire** à Claude d’inventer les run_id.

---

## 10. Partie résultats — faits réels à utiliser

### 10.1. Run A — baseline
Configuration réelle vérifiée :

- modèle : `SiameseResNet18`
- `pretrained=True`
- `dropout=0.2`
- `loss=focal`
- `focal_gamma=3.0`
- `learning_rate=1e-4`
- `weight_decay=1e-3`
- `batch_size=16`
- `epochs planifiées=30`
- `epochs effectives=26`

Meilleure validation :

- meilleur `val_macro_f1 = 0.4778` à l’époque **19**
- meilleure `val_accuracy = 0.8726` à l’époque **26**
- plus faible `val_loss = 0.3218` à l’époque **13**

Test :

- `test_loss = 0.5156`
- `test_accuracy = 0.7677`
- `test_macro_f1 = 0.5558`
- `test_weighted_f1 = 0.8365`
- `test_macro_precision = 0.5528`
- `test_macro_recall = 0.6304`

Matrice de confusion test :

- true `no-damage` -> `608, 161, 4, 4`
- true `minor-damage` -> `13, 10, 2, 0`
- true `major-damage` -> `4, 2, 15, 0`
- true `destroyed` -> `0, 0, 3, 5`

Lecture analytique attendue :

- baseline fonctionnelle
- surapprentissage fort
- confusion majeure entre `no-damage` et `minor-damage`

### 10.2. Run B — modèle régularisé final
Configuration réelle vérifiée :

- modèle : `SiameseResNet18`
- `pretrained=True`
- `dropout=0.4`
- `loss=focal`
- `focal_gamma=2.0`
- `learning_rate=1e-4`
- `weight_decay=5e-3`
- `batch_size=16`
- `epochs planifiées=30`
- `epochs effectives=30`

Meilleure validation :

- meilleur `val_macro_f1 = 0.5963` à l’époque **29**
- meilleure `val_accuracy = 0.9231` à l’époque **29**
- plus faible `val_loss = 0.4540` à l’époque **22**

Test :

- `test_loss = 0.3612`
- `test_accuracy = 0.9001`
- `test_macro_f1 = 0.5812`
- `test_weighted_f1 = 0.9127`
- `test_macro_precision = 0.5741`
- `test_macro_recall = 0.6136`

Matrice de confusion test :

- true `no-damage` -> `727, 45, 2, 3`
- true `minor-damage` -> `20, 3, 2, 0`
- true `major-damage` -> `6, 2, 11, 2`
- true `destroyed` -> `0, 0, 1, 7`

Lecture analytique attendue :

- meilleur modèle global
- forte amélioration sur `no-damage`
- forte amélioration sur `destroyed`
- `minor-damage` reste difficile

### 10.3. Run C — BIT Transformer
Configuration réelle vérifiée :

- modèle : `BITTransformerRunC`
- `loss=focal`
- `focal_gamma=2.0`
- `learning_rate=3e-5`
- `weight_decay=1e-2`
- `dropout=0.2`
- `attention_dropout=0.1`
- `batch_size=16`
- `epochs planifiées=40`
- `epochs effectives=11`

Meilleure validation :

- meilleur `val_macro_f1 = 0.1786` à l’époque **3**

Test :

- `test_loss = 3.8090`
- `test_accuracy = 0.0421`
- `test_macro_f1 = 0.1731`
- `test_weighted_f1 = 0.0174`
- `test_macro_precision = 0.1640`
- `test_macro_recall = 0.5004`

Matrice de confusion test :

- true `no-damage` -> `0, 393, 6, 378`
- true `minor-damage` -> `0, 17, 0, 8`
- true `major-damage` -> `0, 4, 12, 5`
- true `destroyed` -> `0, 0, 2, 6`

Lecture analytique attendue :

- échec de généralisation
- très faible adaptation au jeu de données
- non prédiction correcte de la classe `no-damage`

---

## 11. Partie analyse comparative — message scientifique attendu

Claude doit conclure clairement :

- **Run B > Run A > Run C**

### 11.1. Pourquoi Run B est sélectionné
À faire ressortir :

- meilleure accuracy globale
- meilleure macro-F1
- meilleure weighted-F1
- meilleure calibration via test loss plus faible
- meilleure robustesse sur les classes clés

### 11.2. Pourquoi Run A reste utile
Run A doit être présenté comme :

- la baseline de référence
- la preuve que la pipeline fonctionne
- le point de départ pour diagnostiquer le surapprentissage

### 11.3. Pourquoi Run C échoue
Causes attendues dans l’analyse :

- architecture plus exigeante en données
- entraînement from scratch
- dataset trop petit
- déséquilibre extrême
- absence de pré-entraînement Transformer massif

---

## 12. Partie interprétation du surapprentissage

Il faut imposer une analyse train vs val.

### Run A
Dernière époque :

- `train_macro_f1 = 0.9547`
- `val_macro_f1 = 0.4360`
- gap = **0.5186**

### Run B
Dernière époque :

- `train_macro_f1 = 0.9697`
- `val_macro_f1 = 0.5620`
- gap = **0.4077**

### Run C
Dernière époque :

- `train_macro_f1 = 0.5218`
- `val_macro_f1 = 0.0535`
- gap = **0.4683**

Interprétation à exiger :

- Run A surapprend fortement
- Run B surapprend encore, mais généralise mieux
- Run C n’apprend pas correctement une représentation transférable

Important :

- la validation est utile pour piloter l’entraînement
- mais le **test indépendant reste l’arbitre final**

---

## 13. Ce qu’il faut dire sur la sélection du modèle final

Le modèle final est :

- **Run B — Siamese ResNet18 regularized**

Il faut expliquer que ce choix est justifié à la fois par :

- les métriques de test
- la cohérence des matrices de confusion
- la robustesse du comportement global
- la compatibilité avec tout le reste du projet
  - Grad-CAM
  - registry
  - FastAPI
  - Streamlit

---

## 14. Ce qu’il faut dire sur la suite MLOps

Après la phase d’entraînement et de sélection :

- évaluation test standardisée
- génération de rapports
- Grad-CAM sur Run B
- enregistrement dans MLflow Model Registry
- serving FastAPI
- démo Streamlit

Message à faire passer :

- le projet ne s’arrête pas à “entraîner un modèle”
- il suit un **cycle de vie MLOps complet**

---

## 15. Ce qu’il ne faut surtout pas écrire

Interdire explicitement les erreurs suivantes :

- dire que Run A et Run B sont `pretrained=False`
- dire que Run A a la meilleure validation à `0.5963`
- dire que Run B s’est arrêté à 26 époques
- dire que l’augmentation de données “corrige” directement le déséquilibre
- dire que le dossier `mlruns` local confirme sans ambiguïté tous les run_id finaux
- présenter le BIT comme meilleur ou équivalent
- confondre validation et test

---

## 16. Prompt prêt à donner à Claude/Cowork

Tu peux copier-coller le prompt ci-dessous.

```text
Rédige une section académique complète et factuelle de mémoire PFE sur la partie entraînement, expérimentation et évaluation du projet flood_damage_xai_mlops.

Contraintes absolues :
- N’invente aucune information.
- Si une information n’est pas vérifiée, écris [à compléter].
- N’invente jamais de run_id MLflow.
- Distingue toujours métriques de validation et métriques de test.
- Reste cohérent avec un mémoire scientifique de Computer Vision / MLOps.

Objectif du texte :
- expliquer chronologiquement la logique expérimentale du projet ;
- présenter le backbone ResNet18 puis le modèle Siamese ResNet18 ;
- décrire la pipeline de données bi-temporelles ;
- expliquer la data augmentation et la gestion du déséquilibre ;
- documenter les runs A, B et C ;
- analyser les résultats ;
- justifier la sélection finale du run B ;
- relier le tout au cadre MLOps (MLflow Tracking, artefacts, Model Registry).

Informations factuelles à respecter :

1. Dataset
- xBD flooding subset
- 8 471 bâtiments
- splits : train 6313, val 1327, test 831
- classes totales :
  - no-damage 8128
  - minor-damage 149
  - major-damage 119
  - destroyed 75

2. Architecture du modèle final A/B
- backbone : ResNet18
- couche FC retirée et remplacée par Identity
- sortie backbone : 512-D
- fusion siamoise :
  - pre_features
  - post_features
  - abs(post_features - pre_features)
- vecteur fusionné : 1536-D
- tête :
  - Linear(1536 -> 512)
  - ReLU
  - Dropout
  - Linear(512 -> 4)
- paramètres :
  - backbone 11,176,512
  - classifier 788,996
  - total 11,965,508

3. Prétraitement / augmentation
- Resize 224x224
- HorizontalFlip p=0.5
- VerticalFlip p=0.2
- RandomRotate90 p=0.3
- ShiftScaleRotate shift=0.03 scale=0.05 rotate=15 p=0.3
- RandomBrightnessContrast p=0.3
- GaussNoise p=0.2
- Normalize ImageNet
- ToTensorV2
- géométrie synchronisée entre PRE et POST
- photométrie indépendante par branche

4. Gestion du déséquilibre
- WeightedRandomSampler
- class weights
- FocalLoss

5. Run A
- SiameseResNet18
- pretrained=True
- dropout=0.2
- focal gamma=3.0
- lr=1e-4
- wd=1e-3
- batch_size=16
- epochs effectives=26
- meilleur val_macro_f1=0.4778 à l’époque 19
- test :
  - loss 0.5156
  - accuracy 0.7677
  - macro_f1 0.5558
  - weighted_f1 0.8365
  - macro_precision 0.5528
  - macro_recall 0.6304
- confusion test :
  - no-damage -> 608,161,4,4
  - minor-damage -> 13,10,2,0
  - major-damage -> 4,2,15,0
  - destroyed -> 0,0,3,5

6. Run B
- SiameseResNet18 regularized
- pretrained=True
- dropout=0.4
- focal gamma=2.0
- lr=1e-4
- wd=5e-3
- batch_size=16
- epochs effectives=30
- meilleur val_macro_f1=0.5963 à l’époque 29
- test :
  - loss 0.3612
  - accuracy 0.9001
  - macro_f1 0.5812
  - weighted_f1 0.9127
  - macro_precision 0.5741
  - macro_recall 0.6136
- confusion test :
  - no-damage -> 727,45,2,3
  - minor-damage -> 20,3,2,0
  - major-damage -> 6,2,11,2
  - destroyed -> 0,0,1,7

7. Run C
- BIT Transformer
- lr=3e-5
- wd=1e-2
- dropout=0.2
- attention_dropout=0.1
- epochs effectives=11
- meilleur val_macro_f1=0.1786 à l’époque 3
- test :
  - loss 3.8090
  - accuracy 0.0421
  - macro_f1 0.1731
  - weighted_f1 0.0174
  - macro_precision 0.1640
  - macro_recall 0.5004

8. Analyse attendue
- expliquer ResNet18
- expliquer residual learning
- expliquer le Siamese Network
- expliquer la fusion temporelle
- montrer que Run A sert de baseline
- montrer que Run B améliore la généralisation
- montrer que Run C échoue dans ce régime de données
- conclure que Run B est le modèle final

9. Partie MLOps
- expliquer que les runs sont tracés avec MLflow
- paramètres, métriques et artefacts sont sauvegardés
- checkpoints, figures, historiques CSV, confusion matrices, rapports de classification
- si run_id non vérifié localement : écrire [à compléter]

Structure attendue :
1. introduction du chapitre
2. cadre MLOps
3. présentation des modèles
4. pipeline de données
5. run A
6. run B
7. run C
8. comparaison globale
9. sélection du modèle final
10. transition vers registry / XAI / serving

Style :
- français académique
- fluide
- rigoureux
- analytique
- sans listes trop courtes
- avec transitions logiques entre sections
```

---

## 17. Résultat attendu

Si Claude suit correctement ce guide, il doit produire :

- un chapitre cohérent ;
- chronologiquement correct ;
- aligné avec les artefacts réels du projet ;
- compatible avec les résultats finaux ;
- sans contradiction entre architecture, entraînement, résultats et MLOps.
