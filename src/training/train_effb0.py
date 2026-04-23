from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader

from src.data.dataloader import (
    DataLoadersBundle,
    build_split_dataset,
    build_weighted_random_sampler,
    compute_class_weights_from_dataframe,
    filter_split_dataframe,
    load_split_metadata,
)
from src.data.transforms_aggressive import build_aggressive_train_transforms, build_val_transforms
from src.models.siamese_efficientnet_b0 import SiameseEfficientNetB0
from src.training.losses import build_loss
from src.training.train import (
    EarlyStopping,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Siamese EfficientNet-B0 flood damage classifier."
    )
    parser.add_argument(
        "--split-metadata-path",
        "--metadata-csv",
        dest="split_metadata_path",
        type=str,
        default="data/splits/metadata_splits.csv",
        help="Path to metadata_splits.csv. --metadata-csv is kept as a Colab-friendly alias.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/effb0")
    parser.add_argument("--history-path", type=str, default="outputs/history/training_history_effb0.csv")
    parser.add_argument("--figures-dir", type=str, default="outputs/figures/effb0")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="mlruns")
    parser.add_argument("--mlflow-experiment-name", type=str, default="flood_damage_xai_mlops")
    parser.add_argument("--mlflow-run-name", type=str, default="run_c_effb0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--loss-type", type=str, choices=["ce", "focal"], default="focal")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--lr-scheduler", type=str, choices=["none", "cosine", "plateau"], default="cosine")
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=2)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=0.3)
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



def build_effb0_mlflow_params(args: argparse.Namespace, device: torch.device) -> dict[str, str | int | float | bool]:
    return {
        "model_name": "SiameseEfficientNetB0",
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
    }



def build_effb0_dataloaders(
    split_metadata_path: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    random_state: int,
    dataset_root: str | Path | None,
) -> DataLoadersBundle:
    split_df = load_split_metadata(split_metadata_path)
    transforms = {
        "train": build_aggressive_train_transforms(image_size=image_size),
        "val": build_val_transforms(image_size=image_size),
        "test": build_val_transforms(image_size=image_size),
    }

    datasets: dict[str, object] = {}
    for split_name in ("train", "val", "test"):
        current_split_df = filter_split_dataframe(split_df, split_name)
        datasets[split_name] = build_split_dataset(
            split_df=current_split_df,
            metadata_csv=split_metadata_path,
            split=split_name,
            transforms=transforms[split_name],
            return_metadata=True,
            dataset_root=dataset_root,
        )

    num_classes = int(split_df["class_id"].max()) + 1
    class_weights = compute_class_weights_from_dataframe(
        datasets["train"].df,
        label_column="class_id",
        num_classes=num_classes,
    )
    train_sampler = build_weighted_random_sampler(
        datasets["train"].df,
        class_weights=class_weights,
        label_column="class_id",
        random_state=random_state,
    )

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return DataLoadersBundle(
        train_dataset=datasets["train"],
        val_dataset=datasets["val"],
        test_dataset=datasets["test"],
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_sampler=train_sampler,
        class_weights=class_weights,
    )



def fit(args: argparse.Namespace) -> None:
    set_seed(args.random_state)
    device = resolve_device(args.device)
    print(f"Using device for EfficientNet-B0: {device}")

    mlflow.set_tracking_uri(resolve_mlflow_tracking_uri(args.mlflow_tracking_uri))
    mlflow.set_experiment(args.mlflow_experiment_name)

    with mlflow.start_run(run_name=args.mlflow_run_name):
        active_run = mlflow.active_run()
        if active_run is not None:
            print(f"MLflow run_id: {active_run.info.run_id}")

        mlflow.log_params(build_effb0_mlflow_params(args, device))

        dataloaders = build_effb0_dataloaders(
            split_metadata_path=args.split_metadata_path,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            random_state=args.random_state,
            dataset_root=args.dataset_root,
        )

        model = SiameseEfficientNetB0(
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
            best_checkpoint_path = Path(args.output_dir) / "best_siamese_effb0.pt"
            print(f"Resumed EfficientNet-B0 training from checkpoint: {args.resume_from}")

        class_names = build_class_names(args.num_classes)
        stopped_early = False

        for epoch in range(start_epoch, args.epochs + 1):
            train_result = train_one_epoch(
                model=model,
                dataloader=dataloaders.train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                num_classes=args.num_classes,
                scaler=scaler,
                max_batches=args.max_train_batches,
            )
            val_result = evaluate(
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
                f"EfficientNet-B0 Epoch {epoch:03d}/{args.epochs:03d} | "
                f"lr={learning_rate:.8f} "
                f"train_loss={train_result.loss:.4f} train_accuracy={train_result.accuracy:.4f} "
                f"train_macro_f1={train_result.macro_f1:.4f} | "
                f"val_loss={val_result.loss:.4f} val_accuracy={val_result.accuracy:.4f} "
                f"val_macro_f1={val_result.macro_f1:.4f}"
            )

            improved = early_stopping.step(val_macro_f1=val_result.macro_f1, val_loss=val_result.loss)
            if improved:
                best_val_macro_f1 = val_result.macro_f1
                best_val_accuracy = val_result.accuracy
                best_val_loss = val_result.loss
                best_checkpoint_path = save_checkpoint(
                    output_dir=args.output_dir,
                    checkpoint_name="best_siamese_effb0.pt",
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
                print(f"Saved best EfficientNet-B0 checkpoint: {best_checkpoint_path}")

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
                print(f"Early stopping triggered for EfficientNet-B0 at epoch {epoch:03d}.")
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

        print(f"Best EfficientNet-B0 validation macro F1: {best_val_macro_f1:.4f}")
        if best_checkpoint_path is not None:
            print(f"Best EfficientNet-B0 checkpoint path: {best_checkpoint_path}")
        print(f"Last EfficientNet-B0 checkpoint path: {last_checkpoint_path}")
        print(f"Final history CSV: {history_path}")
        print("Final training curves:", ", ".join(str(path) for path in curve_paths.values()))
        print(f"MLflow artifact URI: {mlflow.get_artifact_uri()}")



def main() -> None:
    args = parse_args()
    fit(args)


if __name__ == "__main__":
    main()
