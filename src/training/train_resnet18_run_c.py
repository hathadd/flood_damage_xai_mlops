from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import XBDPairBuildingDataset
from src.data.transforms_run_c import get_run_c_eval_transforms, get_run_c_train_transforms
from src.models.siamese_model import SiameseResNet18
from src.training.losses import build_loss
from src.training.train import (
    EarlyStopping,
    EpochResult,
    build_class_names,
    build_history_row,
    build_scheduler,
    evaluate,
    get_current_learning_rate,
    load_training_checkpoint,
    log_epoch_metrics_to_mlflow,
    resolve_device,
    resolve_mlflow_tracking_uri,
    restore_model_weights,
    save_checkpoint,
    save_confusion_matrix_figure,
    save_history_csv,
    save_training_curves,
    set_seed,
    train_one_epoch,
)

DEFAULT_SPLIT_METADATA_PATH = "data/splits/metadata_splits.csv"
DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 0
DEFAULT_PIN_MEMORY = False
DEFAULT_RANDOM_STATE = 42
DEFAULT_CONTEXT_RATIO = 0.25
DEFAULT_MIN_CROP_SIZE = 64


@dataclass
class RunCDataLoadersBundle:
    train_dataset: XBDPairBuildingDataset
    val_dataset: XBDPairBuildingDataset
    test_dataset: XBDPairBuildingDataset
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_sampler: WeightedRandomSampler
    class_weights: torch.Tensor



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the official Run C Siamese ResNet18 flood damage classifier."
    )
    parser.add_argument(
        "--split-metadata-path",
        "--metadata-csv",
        dest="split_metadata_path",
        type=str,
        default=DEFAULT_SPLIT_METADATA_PATH,
        help="Path to metadata_splits.csv. --metadata-csv is kept as a Colab-friendly alias.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/run_c/checkpoints")
    parser.add_argument("--history-path", type=str, default="outputs/run_c/history/training_history.csv")
    parser.add_argument("--figures-dir", type=str, default="outputs/run_c/figures")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="mlruns")
    parser.add_argument("--mlflow-experiment-name", type=str, default="flood_damage_xai_mlops")
    parser.add_argument("--mlflow-run-name", type=str, default="run_c_resnet18")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=7e-5)
    parser.add_argument("--weight-decay", type=float, default=7e-3)
    parser.add_argument("--loss-type", type=str, choices=["ce", "focal"], default="focal")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--lr-scheduler", type=str, choices=["none", "cosine", "plateau"], default="cosine")
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=2)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--mixed-precision", dest="mixed_precision", action="store_true")
    parser.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    parser.set_defaults(mixed_precision=True)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Optional dataset root override used to resolve relative paths from metadata CSVs.",
    )
    return parser.parse_args()



def load_split_metadata(split_metadata_path: str | Path = DEFAULT_SPLIT_METADATA_PATH) -> pd.DataFrame:
    split_metadata_path = Path(split_metadata_path)
    if not split_metadata_path.exists():
        raise FileNotFoundError(f"Split metadata file not found: {split_metadata_path}")

    df = pd.read_csv(split_metadata_path)
    required_columns = {"split", "class_id", "damage_class", "sample_id"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in split metadata: {missing}")

    return df



def filter_split_dataframe(df: pd.DataFrame, split: str) -> pd.DataFrame:
    split_df = df.loc[df["split"] == split].copy()
    if split_df.empty:
        raise ValueError(f"No samples found for split='{split}'")
    return split_df.reset_index(drop=True)



def build_split_dataset(
    split_df: pd.DataFrame,
    metadata_csv: str | Path,
    split: str,
    transforms: object,
    dataset_root: str | Path | None,
    context_ratio: float = DEFAULT_CONTEXT_RATIO,
    min_crop_size: int = DEFAULT_MIN_CROP_SIZE,
) -> XBDPairBuildingDataset:
    dataset = XBDPairBuildingDataset(
        metadata_csv=metadata_csv,
        split=split,
        transforms=transforms,
        context_ratio=context_ratio,
        min_crop_size=min_crop_size,
        return_metadata=True,
        dataset_root=dataset_root,
    )
    dataset.df = split_df.reset_index(drop=True)
    return dataset



def compute_class_weights_from_dataframe(
    df: pd.DataFrame,
    label_column: str = "class_id",
    num_classes: int | None = None,
) -> torch.Tensor:
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in dataframe.")

    label_counts = df[label_column].value_counts().sort_index()
    if label_counts.empty:
        raise ValueError("Cannot compute class weights from an empty dataframe.")

    if num_classes is None:
        num_classes = int(label_counts.index.max()) + 1

    missing_classes = [class_index for class_index in range(num_classes) if class_index not in label_counts.index]
    if missing_classes:
        raise ValueError(
            f"Train split is missing classes required for weighting: {missing_classes}"
        )

    total_samples = int(label_counts.sum())
    class_weights = torch.zeros(num_classes, dtype=torch.float32)
    for class_index in range(num_classes):
        class_count = int(label_counts.loc[class_index])
        class_weights[class_index] = total_samples / (num_classes * class_count)
    return class_weights



def build_weighted_random_sampler(
    train_df: pd.DataFrame,
    class_weights: torch.Tensor,
    label_column: str = "class_id",
    random_state: int = DEFAULT_RANDOM_STATE,
) -> WeightedRandomSampler:
    sample_weights = class_weights[train_df[label_column].to_numpy()].double()
    generator = torch.Generator()
    generator.manual_seed(random_state)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )



def build_run_c_dataloaders(
    split_metadata_path: str | Path = DEFAULT_SPLIT_METADATA_PATH,
    image_size: int = DEFAULT_IMAGE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    pin_memory: bool = DEFAULT_PIN_MEMORY,
    random_state: int = DEFAULT_RANDOM_STATE,
    dataset_root: str | Path | None = None,
) -> RunCDataLoadersBundle:
    split_df = load_split_metadata(split_metadata_path)
    train_transforms = get_run_c_train_transforms(image_size=image_size)
    eval_transforms = get_run_c_eval_transforms(image_size=image_size)

    train_df = filter_split_dataframe(split_df, "train")
    val_df = filter_split_dataframe(split_df, "val")
    test_df = filter_split_dataframe(split_df, "test")

    train_dataset = build_split_dataset(
        split_df=train_df,
        metadata_csv=split_metadata_path,
        split="train",
        transforms=train_transforms,
        dataset_root=dataset_root,
    )
    val_dataset = build_split_dataset(
        split_df=val_df,
        metadata_csv=split_metadata_path,
        split="val",
        transforms=eval_transforms,
        dataset_root=dataset_root,
    )
    test_dataset = build_split_dataset(
        split_df=test_df,
        metadata_csv=split_metadata_path,
        split="test",
        transforms=eval_transforms,
        dataset_root=dataset_root,
    )

    num_classes = int(split_df["class_id"].max()) + 1
    class_weights = compute_class_weights_from_dataframe(
        train_dataset.df,
        label_column="class_id",
        num_classes=num_classes,
    )
    train_sampler = build_weighted_random_sampler(
        train_dataset.df,
        class_weights=class_weights,
        label_column="class_id",
        random_state=random_state,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return RunCDataLoadersBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_sampler=train_sampler,
        class_weights=class_weights,
    )



def build_run_c_mlflow_params(args: argparse.Namespace, device: torch.device) -> dict[str, str | int | float | bool]:
    return {
        "model_name": "SiameseResNet18",
        "run_profile": "run_c",
        "split_metadata_path": args.split_metadata_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "pretrained": args.pretrained,
        "num_classes": args.num_classes,
        "random_state": args.random_state,
        "device": str(device),
        "loss_type": args.loss_type,
        "focal_gamma": args.focal_gamma,
        "lr_scheduler": args.lr_scheduler,
        "min_learning_rate": args.min_learning_rate,
        "plateau_factor": args.plateau_factor,
        "plateau_patience": args.plateau_patience,
        "warmup_epochs": args.warmup_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "mixed_precision": args.mixed_precision,
        "resume_from": args.resume_from or "",
        "dataset_root": args.dataset_root or "",
        "transform_profile": "run_c_stronger_regularization",
    }



def fit(args: argparse.Namespace) -> None:
    set_seed(args.random_state)
    device = resolve_device(args.device)
    print(f"Using device for Run C: {device}")

    mlflow.set_tracking_uri(resolve_mlflow_tracking_uri(args.mlflow_tracking_uri))
    mlflow.set_experiment(args.mlflow_experiment_name)

    with mlflow.start_run(run_name=args.mlflow_run_name):
        active_run = mlflow.active_run()
        if active_run is not None:
            print(f"MLflow run_id: {active_run.info.run_id}")

        mlflow.log_params(build_run_c_mlflow_params(args, device))

        dataloaders = build_run_c_dataloaders(
            split_metadata_path=args.split_metadata_path,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            random_state=args.random_state,
            dataset_root=args.dataset_root,
        )

        model = SiameseResNet18(
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            dropout=args.dropout,
        ).to(device)
        criterion = build_loss(
            loss_type=args.loss_type,
            class_weights=dataloaders.class_weights.to(device),
            gamma=args.focal_gamma,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = build_scheduler(
            optimizer=optimizer,
            scheduler_name=args.lr_scheduler,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            min_learning_rate=args.min_learning_rate,
            plateau_factor=args.plateau_factor,
            plateau_patience=args.plateau_patience,
        )
        use_mixed_precision = args.mixed_precision and device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)
        early_stopping = EarlyStopping(patience=args.early_stopping_patience)

        history: list[dict[str, float | int]] = []
        start_epoch = 1
        best_val_macro_f1 = -1.0
        best_val_accuracy = 0.0
        best_val_loss = float("inf")
        best_checkpoint_path: Path | None = None
        last_checkpoint_path = Path(args.output_dir) / "last_epoch.pt"

        if args.resume_from is not None:
            checkpoint = load_training_checkpoint(
                checkpoint_path=args.resume_from,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                early_stopping=early_stopping,
                device=device,
            )
            history = list(checkpoint.get("history", []))
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_val_macro_f1 = float(
                checkpoint.get("best_val_macro_f1", checkpoint.get("val_macro_f1", best_val_macro_f1))
            )
            best_val_accuracy = float(
                checkpoint.get("best_val_accuracy", checkpoint.get("val_accuracy", best_val_accuracy))
            )
            best_val_loss = float(
                checkpoint.get("best_val_loss", checkpoint.get("val_loss", best_val_loss))
            )
            best_checkpoint_path = Path(args.output_dir) / "best_siamese_resnet18_run_c.pt"
            print(f"Resumed Run C training from checkpoint: {args.resume_from}")

        class_names = build_class_names(args.num_classes)
        stopped_early = False

        for epoch in range(start_epoch, args.epochs + 1):
            train_result: EpochResult = train_one_epoch(
                model=model,
                dataloader=dataloaders.train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                num_classes=args.num_classes,
                scaler=scaler,
                max_batches=args.max_train_batches,
            )
            val_result: EpochResult = evaluate(
                model=model,
                dataloader=dataloaders.val_loader,
                criterion=criterion,
                device=device,
                num_classes=args.num_classes,
                mixed_precision=use_mixed_precision,
                max_batches=args.max_val_batches,
            )

            confusion_matrix_path = save_confusion_matrix_figure(
                matrix=val_result.confusion_matrix,
                class_names=class_names,
                output_path=Path(args.figures_dir) / "confusion_matrix" / f"epoch_{epoch:03d}.png",
            )
            mlflow.log_artifact(str(confusion_matrix_path), artifact_path="figures/confusion_matrix")
            learning_rate = get_current_learning_rate(optimizer)

            history.append(build_history_row(epoch, train_result, val_result, learning_rate))
            log_epoch_metrics_to_mlflow(epoch, train_result, val_result, learning_rate)
            scheduler.step(val_loss=val_result.loss)

            print(
                f"Run C Epoch {epoch:03d}/{args.epochs:03d} | "
                f"lr={learning_rate:.8f} "
                f"train_loss={train_result.loss:.4f} "
                f"train_macro_f1={train_result.macro_f1:.4f} | "
                f"val_loss={val_result.loss:.4f} "
                f"val_macro_f1={val_result.macro_f1:.4f}"
            )

            improved = early_stopping.step(val_macro_f1=val_result.macro_f1, val_loss=val_result.loss)
            if improved:
                best_val_macro_f1 = val_result.macro_f1
                best_val_accuracy = val_result.accuracy
                best_val_loss = val_result.loss
                best_checkpoint_path = save_checkpoint(
                    output_dir=args.output_dir,
                    checkpoint_name="best_siamese_resnet18_run_c.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    val_result=val_result,
                    args=args,
                    history=history,
                    best_val_macro_f1=best_val_macro_f1,
                    best_val_accuracy=best_val_accuracy,
                    best_val_loss=best_val_loss,
                    early_stopping=early_stopping,
                )
                print(f"Saved best Run C checkpoint: {best_checkpoint_path}")

            last_checkpoint_path = save_checkpoint(
                output_dir=args.output_dir,
                checkpoint_name="last_epoch.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                val_result=val_result,
                args=args,
                history=history,
                best_val_macro_f1=best_val_macro_f1,
                best_val_accuracy=best_val_accuracy,
                best_val_loss=best_val_loss,
                early_stopping=early_stopping,
            )

            if early_stopping.should_stop:
                stopped_early = True
                print(f"Early stopping triggered for Run C at epoch {epoch:03d}.")
                if best_checkpoint_path is not None and best_checkpoint_path.exists():
                    restore_model_weights(best_checkpoint_path, model, device)
                break

        history_path = save_history_csv(history, args.history_path)
        curve_paths = save_training_curves(history, args.figures_dir)

        mlflow.log_metrics(
            {
                "best_val_macro_f1": best_val_macro_f1,
                "best_val_accuracy": best_val_accuracy,
                "best_val_loss": best_val_loss,
                "stopped_early": float(stopped_early),
            }
        )
        mlflow.log_artifact(str(history_path), artifact_path="history")
        for curve_path in curve_paths.values():
            mlflow.log_artifact(str(curve_path), artifact_path="figures")
        if best_checkpoint_path is not None and best_checkpoint_path.exists():
            mlflow.log_artifact(str(best_checkpoint_path), artifact_path="checkpoints")
        if last_checkpoint_path.exists():
            mlflow.log_artifact(str(last_checkpoint_path), artifact_path="checkpoints")

        print(f"Best Run C validation macro F1: {best_val_macro_f1:.4f}")
        if best_checkpoint_path is not None:
            print(f"Best Run C checkpoint path: {best_checkpoint_path}")
        print(f"Last Run C checkpoint path: {last_checkpoint_path}")
        print(f"Final history CSV: {history_path}")
        print("Final training curves:", ", ".join(str(path) for path in curve_paths.values()))
        print(f"MLflow artifact URI: {mlflow.get_artifact_uri()}")



def main() -> None:
    args = parse_args()
    fit(args)


if __name__ == "__main__":
    main()
