# Training Pipeline Explained

This document is a guided map of the training side of `flood_damage_xai_mlops`. It is meant to help you understand the project slowly, concept by concept, before changing anything.

## A. Data Foundation

### Dataset root
- The project uses a dataset root directory that contains the xBD flooding subset imagery and labels.
- In the current config, the local default root is defined in `configs/data.yaml` under `dataset.root_dir`.
- On Colab, the training and evaluation scripts override that root with `--dataset-root /content/drive/MyDrive/flooding_dataset`.

### `metadata.csv`
- Canonical path: `data/interim/metadata.csv`
- Role: this is the building-level metadata table produced from the original xBD files.
- Each row represents one building sample, not one whole image tile.
- Important columns include:
  - `sample_id`
  - `building_uid`
  - `pre_image_path`
  - `post_image_path`
  - `damage_class`
  - `class_id`
  - `wkt`
  - `image_width`
  - `image_height`
- The paths stored in the CSV are relative paths such as `images/pre/...`, not machine-specific absolute paths.

### `metadata_splits.csv`
- Canonical path: `data/splits/metadata_splits.csv`
- Role: this is the split-aware version used by training and evaluation.
- It adds a `split` column with `train`, `val`, and `test`.
- Current columns are:
  - `sample_id`, `building_uid`, `disaster`, `disaster_type`, `capture_date`, `sensor`
  - `image_width`, `image_height`
  - `pre_image_name`, `post_image_name`
  - `pre_image_path`, `post_image_path`, `label_json_path`
  - `damage_class`, `wkt`, `class_id`
  - `pre_exists`, `post_exists`, `label_exists`
  - `split`

### Train / val / test split
Current sample counts:
- `train = 6313`
- `val = 1327`
- `test = 831`

The project trains on building-level rows, but split assignment is stored directly in `metadata_splits.csv` so every training or evaluation script can filter consistently.

### Class imbalance
Current class distribution across the whole split file:
- `no-damage = 8128`
- `minor-damage = 149`
- `major-damage = 119`
- `destroyed = 75`

This is a severe imbalance problem. The model can get a high accuracy by predicting the majority class too often, which is why macro F1 matters so much in this project.

Per split:
- Train:
  - `no-damage = 6090`
  - `minor-damage = 96`
  - `major-damage = 80`
  - `destroyed = 47`
- Val:
  - `no-damage = 1261`
  - `minor-damage = 28`
  - `major-damage = 18`
  - `destroyed = 20`
- Test:
  - `no-damage = 777`
  - `minor-damage = 25`
  - `major-damage = 21`
  - `destroyed = 8`

### How `sample_id` and `building_uid` are used
- `sample_id` identifies the original disaster image pair / scene context.
- `building_uid` identifies a specific building polygon inside that sample.
- One `sample_id` can contain many buildings, so many rows can share the same `sample_id`.
- `building_uid` gives row-level building identity.
- In practice:
  - `sample_id` is useful for grouping and traceability
  - `building_uid` is useful for unique building-level analysis, prediction review, and Grad-CAM output naming

### How pre/post image paths are resolved
- The CSV stores relative paths.
- `XBDPairBuildingDataset` resolves them using `resolve_dataset_root()` and `resolve_data_path()` from `src/data/path_utils.py`.
- This makes the same metadata portable across local/WSL and Colab, as long as `dataset_root` is provided correctly.

## B. Dataset and DataLoader

### Role of `XBDPairBuildingDataset`
File: `src/data/dataset.py`

This is the central building-level dataset class.

Responsibilities:
- read the metadata CSV
- read pre-disaster and post-disaster RGB images
- parse the building polygon WKT
- compute a building crop with context around it
- apply transforms
- return tensors and label information

### What `__getitem__` returns
Each sample returns a dictionary containing at least:
- `pre_image`
- `post_image`
- `label`

When `return_metadata=True`, it also returns:
- `sample_id`
- `building_uid`
- `damage_class`
- `crop_box`
- `pre_image_path`
- `post_image_path`

### Shape of `pre_image` and `post_image` tensors
- For one sample after transforms: `[3, H, W]`
- In practice with the current training/eval pipelines: `[3, 224, 224]`
- For one DataLoader batch: `[B, 3, 224, 224]`

### Difference between Dataset and DataLoader
- `Dataset` defines how to read one sample.
- `DataLoader` defines how to batch many samples together, iterate efficiently, and optionally sample them in a special way.

A simple mental model:
- Dataset = recipe for one item
- DataLoader = machine that serves mini-batches of those items during training

### How batch size works
- `batch_size=16` means the model receives 16 building pairs at once.
- The batch contains:
  - `pre_image` with shape `[16, 3, 224, 224]`
  - `post_image` with shape `[16, 3, 224, 224]`
  - `label` with shape `[16]`
- Larger batch sizes can stabilize gradients but need more GPU memory.

### How DataLoader creates mini-batches
- It repeatedly asks the dataset for single examples.
- Then it stacks tensors into mini-batches.
- For training, the project often uses a sampler instead of plain sequential access.

### Role of `num_workers`
- `num_workers` controls how many subprocesses prepare data in parallel.
- `0` means data loading happens in the main process.
- Higher values can speed up loading, especially on Colab or Linux, but also increase complexity and RAM use.

### Role of `pin_memory`
- `pin_memory=True` can speed up transfer from CPU memory to GPU memory.
- It is helpful mainly when training/evaluating on CUDA.

### How weighted sampling / class balancing is used
File: `src/data/dataloader.py`

For the default Run A / Run B pipeline, the training DataLoader uses:
- `compute_class_weights_from_dataframe()`
- `build_weighted_random_sampler()`

Effect:
- minority classes receive larger sampling weights
- majority class rows are sampled less aggressively
- this does not change the validation or test split
- it only changes how training examples are drawn each epoch

This is one of the project’s main defenses against severe class imbalance.

## C. Data Augmentation

### Why train and validation transforms must be different
- Train transforms should introduce useful randomness so the model generalizes better.
- Validation and test transforms must stay deterministic so performance estimates remain stable and comparable.
- If validation used random augmentation, the metric would depend on randomness and become harder to trust.

### Run A transforms
Files:
- training path: `src/training/train.py`
- dataloaders: `src/data/dataloader.py`
- transform file: `src/data/transforms.py`

Run A train transforms:
- shared geometric transforms on the pair:
  - `Resize(224, 224)`
  - `HorizontalFlip(p=0.5)`
  - `VerticalFlip(p=0.2)`
  - `RandomRotate90(p=0.3)`
  - `ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=15, p=0.3)`
- then separate photometric transforms per branch:
  - `RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3)`
  - `GaussNoise(p=0.2)`
  - `Normalize(...)`
  - `ToTensorV2()`

Run A eval transforms:
- `Resize(224, 224)`
- `Normalize(...)`
- `ToTensorV2()`

Synchronization behavior:
- geometry is synchronized between pre and post
- photometric changes are not synchronized, because `pre` and `post` have separate photometric pipelines

### Run B transforms
Files:
- training path: `src/training/train.py`
- transform file: `src/data/transforms.py`

Run B uses the same transform file and same default augmentation structure as Run A.

The main Run B improvements came from regularization and hyperparameter choices, not from a different transform file.

### Run C ResNet18 augmentation attempt
Files:
- training path: `src/training/train_resnet18_run_c.py`
- transform file: `src/data/transforms_run_c.py`

Run C ResNet train transforms:
- `Resize(224, 224)`
- `HorizontalFlip(p=0.5)`
- `VerticalFlip(p=0.25)`
- `RandomRotate90(p=0.35)`
- `Affine(translate_percent +/- 0.04, scale 0.90 to 1.10, rotate -20 to 20, p=0.45)`
- `RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.45)`
- `HueSaturationValue(p=0.25)`
- `GaussNoise(p=0.15)`
- `CoarseDropout(p=0.20)`
- `Normalize(...)`
- `ToTensorV2()`

Run C ResNet eval transforms:
- `Resize(224, 224)`
- `Normalize(...)`
- `ToTensorV2()`

Synchronization behavior:
- all transforms are synchronized because the full chain is a single `ReplayCompose`
- that means both geometry and photometric changes are replayed on pre and post

### Run C BIT transforms
Files:
- training path: `src/training/train_bit_run_c.py`
- transform file: `src/data/transforms_bit_run_c.py`

Run C BIT train transforms:
- `Resize(224, 224)`
- `HorizontalFlip(p=0.5)`
- `VerticalFlip(p=0.15)`
- `RandomRotate90(p=0.25)`
- `Affine(translate_percent +/- 0.03, scale 0.95 to 1.05, rotate -12 to 12, p=0.35)`
- `RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.35)`
- `GaussNoise(p=0.10)`
- `Normalize(...)`
- `ToTensorV2()`

Run C BIT eval transforms:
- `Resize(224, 224)`
- `Normalize(...)`
- `ToTensorV2()`

Synchronization behavior:
- all transforms are synchronized through `ReplayCompose`

### Which augmentations are random
Random transforms include:
- flips
- rotations
- affine transforms
- brightness / contrast
- hue / saturation adjustments
- noise
- coarse dropout

Deterministic transforms include:
- resize
- normalize
- tensor conversion

### Risks of too much augmentation
Too much augmentation can:
- erase small building damage cues
- create unrealistic satellite artifacts
- destroy the temporal consistency between pre and post
- make minority classes harder to learn because the signal is already weak

This is especially risky for `minor-damage`, where the visual change is subtle.

## D. Model Architectures

### Siamese ResNet18 used in Run A / Run B
File: `src/models/siamese_model.py`

Core idea:
- the same ResNet18 backbone processes pre and post images separately
- this is a Siamese architecture because the two branches share weights

Processing flow:
1. `pre_image` goes through the backbone
2. `post_image` goes through the same backbone
3. two feature vectors are extracted
4. the model computes `abs(post - pre)`
5. the final fused vector is `concat(pre, post, abs_diff)`
6. the classifier head predicts one of the 4 classes

### Feature fusion
The fusion strategy is simple and effective:
- pre features = context before disaster
- post features = context after disaster
- absolute difference = explicit temporal change signal

This is a good match for change-sensitive classification.

### Classifier head
Run A / Run B head:
- `Linear(fused_dim, feature_dim)`
- `ReLU()`
- `Dropout(p=dropout)`
- `Linear(feature_dim, num_classes)`

### Dropout
- Run A used `dropout = 0.2`
- Run B used `dropout = 0.4`
- Higher dropout in Run B is one reason it regularized better.

### Pretrained ImageNet weights
- `pretrained=True` uses `ResNet18_Weights.DEFAULT`
- this gives a strong initialization from ImageNet
- for a small disaster dataset, this is usually much better than training from scratch

### BITTransformerRunC
File: `src/models/bit_transformer_run_c.py`

This is a lightweight bi-temporal Transformer classifier.

Components:
- patch embedding via `Conv2d` with `patch_size=16`
- learnable `CLS` token
- learnable positional embeddings
- shared transformer encoder blocks
- fusion of final branch embeddings

### Patch embedding
- the image is cut into non-overlapping patches
- each patch is projected into an embedding vector
- with `224 x 224` and `patch_size=16`, there are `14 x 14 = 196` patches

### CLS token
- a learnable summary token is prepended to the patch sequence
- after the encoder, the `CLS` token acts as the global representation for the branch

### Positional embedding
- Transformers do not know spatial positions by default
- positional embeddings tell the model where each patch is located in the image

### Transformer encoder
Each encoder block contains:
- `LayerNorm`
- multi-head self-attention
- dropout
- another `LayerNorm`
- MLP block with GELU

### Feature fusion in BIT Run C
Same temporal logic as the Siamese CNN family:
- `concat(z_pre, z_post, abs(z_post - z_pre))`

This keeps the temporal change representation explicit.

### Why BIT failed on this dataset
Run C BIT underperformed badly on the test set.
Most likely reasons:
- the dataset is small for a Transformer
- class imbalance is extreme
- minority classes are very underrepresented
- subtle flood damage cues are hard to learn without stronger inductive bias
- ResNet18 with transfer learning and simpler feature fusion is a better fit for this data scale

The test metrics confirm a collapse:
- `test_accuracy ˜ 0.0421`
- `test_macro_f1 ˜ 0.1731`
- `no-damage` recall dropped to `0.0`

So the theoretically more advanced model was practically a worse fit here.

## E. Training Logic

### Forward pass
For one batch:
1. load `pre_image`, `post_image`, `label`
2. send tensors to device
3. run `model(pre_image, post_image)`
4. obtain logits `[B, num_classes]`

### Loss computation
- The project supports weighted cross-entropy and focal loss.
- Run A, Run B, and Run C BIT all used focal loss in their final recorded checkpoints.
- Class weights are computed from the train split.
- That makes rare classes more important during optimization.

### Backward pass
Inside training:
1. zero gradients
2. compute loss
3. `backward()`
4. gradient clipping with `max_norm=1.0`
5. optimizer step

### Optimizer step
- The main training code uses `AdamW`
- `AdamW` is useful when combining adaptive optimization with explicit weight decay

### Scheduler
- Supported schedulers:
  - `none`
  - `cosine`
  - `plateau`
- The selected runs use cosine scheduling.

### Warmup
- Learning rate warmup starts training with a smaller LR for the first few epochs
- This is especially helpful with pretrained models, focal loss, or mixed precision

### Cosine LR
- After warmup, the learning rate decays smoothly toward `min_learning_rate`
- This often gives more stable convergence than a constant LR

### Mixed precision
- Controlled by `--mixed-precision`
- Uses `torch.cuda.amp`
- Benefits:
  - lower GPU memory usage
  - often faster training on CUDA

### Early stopping
- Implemented in `src/training/train.py`
- The monitored logic is primarily `val_macro_f1`, with `val_loss` used as a tiebreaker
- This is a good choice because macro F1 better reflects imbalanced-class performance

### Checkpoint saving
The training scripts save:
- best checkpoint according to validation improvement
- `last_epoch.pt`
- history CSV
- training curves
- confusion matrices per epoch

### MLflow logging
Each run logs:
- hyperparameters
- epoch metrics
- best validation metrics
- history CSV
- figures
- checkpoints

This makes experiments traceable and comparable.

## F. Hyperparameters Per Run

### Main comparison table

| Run | Model | Loss | Focal gamma | Learning rate | Weight decay | Dropout | Batch size | Epochs | Warmup | Early stopping | Augmentation profile | Checkpoint path | Output path | MLflow run name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Run A | Siamese ResNet18 | focal | 3.0 | 1e-4 | 1e-3 | 0.2 | 16 | 30 | 2 | 7 | `src/data/transforms.py` | `outputs/focal/checkpoints/best_siamese_resnet18.pt` | `outputs/focal/` | `focal_resnet18_gamma3_cosine_warmup2_amp` |
| Run B | Siamese ResNet18 regularized | focal | 2.0 | 1e-4 | 5e-3 | 0.4 | 16 | 30 | 2 | 7 | `src/data/transforms.py` | `outputs/focal_run_b_regularized/checkpoints/best_siamese_resnet18.pt` | `outputs/focal_run_b_regularized/` | `focal_resnet18_run_b_dropout04_wd5e3_gamma2` |
| Run C BIT | BITTransformerRunC | focal | 2.0 | 3e-5 | 1e-2 | 0.2 | 16 | 40 | 5 | 8 | `src/data/transforms_bit_run_c.py` | `outputs/focal_run_c_bit_transformer/checkpoints/best_bit_run_c.pt` | `outputs/focal_run_c_bit_transformer/` | `focal_bit_run_c_transformer_lr3e5_wd1e2` |

### Run C ResNet18 augmentation attempt
This experimental path exists in code but is not the final selected model path.

Main settings from `src/training/train_resnet18_run_c.py`:
- model: Siamese ResNet18
- loss: focal
- gamma: 2.0
- learning rate: `7e-5`
- weight decay: `7e-3`
- dropout: `0.5`
- epochs: `40`
- warmup: `3`
- early stopping: `8`
- transforms: `src/data/transforms_run_c.py`

## G. Metrics

### Accuracy
- Fraction of all correct predictions
- Easy to understand but dangerous under class imbalance

### Precision
- Of the samples predicted as a class, how many were correct?

### Recall
- Of the true samples of a class, how many were found?

### F1-score
- Harmonic mean of precision and recall
- Useful when false positives and false negatives both matter

### Macro F1
- Average F1 across classes, giving each class equal weight
- This is the most important summary metric in this project because rare classes matter scientifically

### Weighted F1
- Average F1 weighted by class support
- Helpful, but still influenced strongly by the majority class

### Confusion matrix
- Shows how classes are confused with each other
- Very useful for understanding whether the model is collapsing toward `no-damage`

### Classification report
- Gives precision, recall, F1, and support for each class
- This is essential for understanding minority-class behavior

### Why macro F1 matters here
Because the dataset is dominated by `no-damage`, a model can look strong by accuracy alone while performing poorly on `minor-damage`, `major-damage`, or `destroyed`.

### Why accuracy alone is misleading
Example:
- If a model predicts `no-damage` almost everywhere, accuracy can still be high.
- But the model would fail the real disaster-assessment purpose.

That is why Run B was selected using a balance of:
- test accuracy
- test macro F1
- class-wise behavior

## H. Results Interpretation

### Final known test metrics
- Run A:
  - `test_accuracy ˜ 0.7677`
  - `test_macro_f1 ˜ 0.5558`
- Run B:
  - `test_accuracy ˜ 0.9001`
  - `test_macro_f1 ˜ 0.5812`
  - `test_weighted_f1 ˜ 0.9127`
- Run C BIT:
  - `test_accuracy ˜ 0.0421`
  - `test_macro_f1 ˜ 0.1731`

### Why Run B was selected
Run B was selected because it achieved the best overall balance:
- much better accuracy than Run A
- better macro F1 than Run A
- far better weighted F1 than Run A
- dramatically better than Run C BIT

Run B improved general performance without sacrificing the imbalanced-class perspective.

### Why `minor-damage` is difficult
`minor-damage` is the hardest class because:
- it has only 25 samples in the test split
- visual differences can be subtle
- it may resemble both `no-damage` and `major-damage`
- even the best model still struggles on that class

Observed on Run B test report:
- `minor-damage` precision = `0.06`
- `minor-damage` recall = `0.12`
- `minor-damage` F1 = `0.08`

That is weak, but still scientifically informative: the task is genuinely difficult at this damage level.

### What the confusion matrix says
At a high level:
- Run B predicts `no-damage` much better than Run A
- Run B improves the `destroyed` class strongly
- `major-damage` remains usable but imperfect
- `minor-damage` remains the main failure point

This is exactly the kind of insight a confusion matrix is meant to reveal.

### Why BIT failed despite being more advanced on paper
A stronger architecture is not always a better practical model.

BIT likely failed because:
- data scale was too small
- imbalance was too severe
- the model had fewer helpful built-in visual priors than ResNet18 transfer learning
- optimization became unstable relative to the available signal

So the project outcome is a good engineering lesson:
- choose the model that works best on your real data, not the model that sounds most advanced

### What Grad-CAM contributes
Grad-CAM adds interpretability after model selection.

It helps answer:
- is the model looking at the building area?
- is it reacting to post-disaster damage cues?
- is it relying on spurious background regions?

This is important for both scientific credibility and future product trust.

## I. MLOps Continuity

The project now forms a coherent pipeline:

1. Data foundation
- build metadata
- build splits
- keep paths portable

2. DVC readiness
- metadata artifacts were cleaned so they can be versioned reproducibly

3. Training runs
- Run A baseline
- Run B regularized final model
- Run C exploratory Transformer

4. Evaluation reports
- unified test evaluation with saved metrics, confusion matrices, and predictions

5. Grad-CAM XAI
- selected model explanations on test samples
- useful for scientific validation and later app UX

6. MLflow Model Registry
- final model can be packaged, versioned, and tagged as the official selected model

7. Future FastAPI serving
- natural next step: load registered model and expose prediction endpoints
- likely flow: input pre/post crops -> prediction -> confidence -> Grad-CAM explanation
- note: serving is a future step; current repo has `src/serving/` but no full deployed API path yet

8. Future monitoring
- once served, the project can add:
  - input drift checks
  - prediction distribution monitoring
  - confidence tracking
  - hard-case review loops

## Final mental model
If you want a simple end-to-end picture, think of the project like this:

- `metadata_splits.csv` says what each building sample is and which split it belongs to
- `XBDPairBuildingDataset` reads the pre/post images and crops the building
- transforms convert crops into model-ready tensors
- DataLoader builds batches
- the model predicts one of four damage classes
- the loss pushes the model to improve
- metrics decide whether the run is actually good on imbalanced data
- evaluation compares runs fairly on the same test split
- Grad-CAM explains the final model
- MLflow Registry packages the final selected model for downstream use
