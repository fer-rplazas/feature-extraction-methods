import argparse
from multiprocessing import Manager
import os
from pathlib import Path
import warnings

import hydra
from joblib import Parallel
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch.cuda import device_count, is_available

from tools.data_processing import Dataset
from tools.nn_training import Module
from tools.patient_data import PatDataset, PatID, Stim, Task
from tools.svm import SVMClassifier

Fs = 2048  # sampling frequency in [Hz]
window_length = 0.5  # Window length in [s]
hop_size = 0.4  # Hop size in [s]. Determines overlap between windows

DEVICE = "gpu" if is_available() else "cpu"


def score_module(module, train_dataloader, valid_dataloader, accelerator, device):

    trainer = pl.Trainer(
        logger=True,
        accelerator=accelerator,
        devices=device,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True,
                mode="max",
                monitor="valid/bal_acc",
                save_top_k=1,
            )
        ],
        max_epochs=150,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="y_pred contains classes not in y_true"
        )
        warnings.filterwarnings("ignore", category=PossibleUserWarning)
        trainer.fit(module, train_dataloader, valid_dataloader)
        module = module.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )
        train_score = trainer.validate(module, train_dataloader)[0]["valid/bal_acc"]
        valid_score = trainer.validate(module, valid_dataloader)[0]["valid/bal_acc"]

    return train_score, valid_score


def score_dataset(
    file,
    cfg,
    arch_hparams: dict,
    save_to_file=True,
    accelerator: str = "gpu",
    device: list[int] | str = [0],
    lock=None,
):

    scores = pd.Series(
        index=[
            "pat",
            "task",
            "stim",
            "fold",
            "svm_train",
            "svm_valid",
            "cnn1d_train",
            "cnn1d_valid",
            "cnn2d_train",
            "cnn2d_valid",
            "ar_train",
            "ar_valid",
        ],
        dtype=object,
    )
    scores[["pat", "task", "stim", "fold"]] = (
        cfg["pat"],
        cfg["task"],
        cfg["stim"],
        cfg["fold"],
    )

    data = PatDataset(cfg["pat"], cfg["task"], cfg["stim"]).load()
    dataset = Dataset(
        data.signals,
        data.label,
        Fs,
        tf_transform=True,
        ar_len=10,
        window_length=window_length,
        hop_size=hop_size,
        fold_id=cfg["fold"],
    )

    # SVM:
    cls = SVMClassifier(
        dataset.X_features_train_scaled,
        dataset.X_features_valid_scaled,
        dataset.y_train,
        dataset.y_valid,
    )

    scores["svm_train"], scores["svm_valid"] = cls.classify()

    # SVM AR:
    cls = SVMClassifier(
        dataset.X_features_ar_train,
        dataset.X_features_ar_valid,
        dataset.y_ar_train,
        dataset.y_ar_valid,
    )

    scores["svm_ar_train"], scores["svm_ar_valid"] = cls.classify()

    # 1D-CNN:
    module = Module.with_defaults_1d(data.signals.shape[0], **arch_hparams.cnn1d)
    scores["cnn1d_train"], scores["cnn1d_valid"] = score_module(
        module,
        dataset.train_dataloader(),
        dataset.valid_dataloader(),
        accelerator,
        device,
    )

    # 2d-CNN:
    module = Module.with_defaults_2d(data.signals.shape[0])
    scores["cnn2d_train"], scores["cnn2d_valid"] = score_module(
        module,
        dataset.train_tf_dataloader(),
        dataset.valid_tf_dataloader(),
        accelerator,
        device,
    )

    # AR Model:
    module = Module.with_defaults_AR(
        data.signals.shape[0], dataset.X_features.shape[1], arch_hparams.cnn1d
    )
    scores["ar_train"], scores["ar_valid"] = score_module(
        module,
        dataset.train_ar_dataloader(),
        dataset.valid_ar_dataloader(),
        accelerator,
        device,
    )

    if save_to_file:
        if lock is not None:
            lock.acquire()
        df = pd.read_csv(file, index_col=0)
        df = pd.concat((df, scores.to_frame().T), ignore_index=True)
        df.to_csv(file)
        if lock is not None:
            lock.release()

    return scores


def prepare_runs(name: str) -> Path:

    filepath = Path("_assets") / Path(name).with_suffix(".csv")

    if not os.path.exists(filepath):
        df = pd.DataFrame(
            columns=[
                "pat",
                "task",
                "stim",
                "fold",
                "svm_train",
                "svm_valid",
                "cnn1d_train",
                "cnn1d_valid",
                "cnn2d_train",
                "cnn2d_valid",
                "ar_train",
                "ar_valid",
            ],
        )

        df.to_csv(filepath)

    return filepath


def sweep_pats(
    name: str,
    arch_hparams: dict,
    n_jobs: int = 4,
    accelerator: str | int = "gpu",
):
    """Prepares database, configures accelerators, and launches parallelized sweep across a range or snrs for the configuration specified in `cfg`"""
    filepath = prepare_runs(name)

    task_list = []
    for pat in PatID:
        for task in Task:
            for stim in Stim:
                if PatDataset(pat, task, stim).exists():
                    [
                        task_list.append(
                            {
                                "pat": pat.name,
                                "task": task.name,
                                "stim": stim.name,
                                "fold": fold,
                            }
                        )
                        for fold in range(5)
                    ]

    if accelerator == "cpu":
        accelerator = "cpu"
        devices = ["auto" for _ in range(len(task_list))]
    elif accelerator == "gpu":  # Spread jobs across available gpus
        accelerator = "gpu"
        devices = np.tile(
            list(range(device_count())), (len(task_list) // device_count()) + 1
        )
        devices = devices[: len(task_list)]
        devices = [[int(x)] for x in devices]
    elif isinstance(accelerator, int):
        accelerator = "gpu"
        devices = [[accelerator] for _ in range(len(task_list))]
    else:
        raise ValueError("`accelerator` not recognized.")

    assert len(devices) == len(task_list), "Problem deciding on gpus for sweep"

    parallel = Parallel(n_jobs=n_jobs)
    mgr = Manager()
    lock = mgr.Lock()
    tasks = []  # (f, *args, **kwargs)
    for task_cfg, device in zip(task_list, devices):
        tasks.append(
            (
                score_dataset,
                (filepath, task_cfg),
                {
                    "arch_hparams": arch_hparams,
                    "accelerator": accelerator,
                    "device": device,
                    "lock": lock,
                },
            )
        )
    _ = parallel(tasks)


def is_int(el: str):
    """Determines whether a str can be converted to int"""
    try:
        int(el)
        return True
    except ValueError:
        return False


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=False, default="patients")
    parser.add_argument("--n_jobs", required=False, default=4, type=int)
    parser.add_argument("--accelerator", required=False, default="gpu", type=str)

    args = parser.parse_args()

    # Load hydra config (model hparams):
    hydra.initialize(version_base=None, config_path="hparams")
    cfg = hydra.compose(config_name="config")

    # Choose compatible accelerator:
    if DEVICE == "cpu":
        if is_int(args.accelerator) or args.accelerator == "gpu":
            warnings.warn("No GPU detected on your system -- using cpu.", UserWarning)
            args.accelerator = "cpu"
    else:
        args.accelerator = (
            int(args.accelerator) if is_int(args.accelerator) else args.accelerator
        )

    sweep_pats(
        args.name,
        n_jobs=args.n_jobs,
        accelerator=args.accelerator,
        arch_hparams=cfg.arch,
    )


if __name__ == "__main__":
    main()
