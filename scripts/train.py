"""
Standalone training entry point (replaces the per-model notebooks for the
revision campaign), so runs can be launched unattended/in-batch and reproduced
exactly from the command line.

Example:
    python scripts/train.py --model LiteMedSeg --dataset CVC-ClinicDB --seed 0 --epochs 100
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation_models import (
    MedicalImageSegmentationData,
    PolypGenSegmentationData,
    SegmentationTrainer,
    TrainConfig,
    create_lightweight_deepsegnet,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_ablation(name: str) -> bool:
    return name.startswith("Abl-")


class _NoAux(torch.nn.Module):
    """LiteMedSeg trained WITHOUT deep supervision: only the main output is returned in training mode
    (the unused auxiliary 1x1 heads receive no gradient). Used for the supervision control (R1.3)."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        return out[0] if isinstance(out, tuple) else out


def build_model(name: str, base_channels: int):
    if name == "LiteMedSeg-noaux":
        return _NoAux(create_lightweight_deepsegnet("tiny", in_channels=3, out_channels=1))
    if is_ablation(name):
        # e.g. Abl-wo_lsk_none_b12 -- see segmentation_models/models/lite_med_seg_ablation.py
        from segmentation_models.models.lite_med_seg_ablation import create_ablation_model
        return create_ablation_model(name)
    if name == "PraNet":
        from segmentation_models.models.pranet import PraNet
        return PraNet(channel=32)
    if name == "LiteMedSeg":
        return create_lightweight_deepsegnet("tiny", in_channels=3, out_channels=1)
    # Configs below match the exact factory calls found in each model's original
    # notebook (e.g. UNet.ipynb), so this rerun changes only the split/metrics/
    # leakage issues under investigation -- not each baseline's capacity.
    if name == "UNet":
        from segmentation_models.models.unet import UNet
        return UNet(n_channels=3, n_classes=1, base_channels=32, bilinear=False)
    if name == "UNet++":
        from segmentation_models.models.unet_pp import NestedUNet
        return NestedUNet(num_classes=1, input_channels=3)
    if name == "ResUNet":
        from segmentation_models.models.res_unet import resunet_small
        return resunet_small(in_channels=3, num_classes=1)
    if name == "ResUNet++":
        from segmentation_models.models.res_unet_pp import create_resunetpp
        return create_resunetpp(config="standard", in_channels=3, num_classes=1)
    if name == "AttentionUNet":
        from segmentation_models.models.attention_unet import attention_unet_small
        return attention_unet_small(in_channels=3, out_channels=1)
    if name == "DeepLabV3+":
        from segmentation_models.models.deeplabv3plus import deeplabv3plus_resnet34
        return deeplabv3plus_resnet34(num_classes=1, pretrained=False, output_stride=16)
    if name == "SCA-InceptionUNet":
        from segmentation_models.models.sca_inception_unet import SCA_InceptionUNeXt
        return SCA_InceptionUNeXt(in_channels=3, out_channels=1)
    if name == "TransUNet":
        from segmentation_models.models.trans_unet import create_transunet
        return create_transunet(config="base", num_classes=1, img_size=256)
    if name == "UNeXt":
        from segmentation_models.models.unext import create_unext
        return create_unext(num_classes=1, in_channels=3)
    if name == "Fast-SCNN":
        from segmentation_models.models.fast_scnn import create_fast_scnn
        return create_fast_scnn(num_classes=1, in_channels=3)
    if name == "MobileViT-Seg":
        from segmentation_models.models.mobilevit_seg import create_mobilevit_seg
        return create_mobilevit_seg(num_classes=1, in_channels=3)
    raise ValueError(f"Unknown model: {name}")


# Models whose forward() can emit a list/tuple of deep-supervision outputs
# depending on construction flags -- all built with those flags off above, so
# BCEDiceLoss (single-tensor) is correct for every entry except LiteMedSeg.
ALL_MODELS = [
    "LiteMedSeg", "UNet", "UNet++", "ResUNet", "ResUNet++",
    "AttentionUNet", "DeepLabV3+", "SCA-InceptionUNet", "TransUNet",
    "UNeXt", "Fast-SCNN", "MobileViT-Seg",
]

# Per-model lr/scheduler exactly as configured in each model's original notebook
# (TrainConfig cell) -- these differ per model and are NOT the TrainConfig
# dataclass defaults, so they must be set explicitly rather than left implicit.
# New (2025-2026) lightweight baselines added for this revision have no such
# notebook to match, so they use LiteMedSeg's recipe (lr=1e-3, CosineAnnealingLR)
# as the most defensible choice available, documented here rather than silently
# defaulting to the dataclass's ReduceLROnPlateau/1e-4.
MODEL_HPARAMS = {
    "LiteMedSeg":        dict(lr=1e-3, scheduler="CosineAnnealingLR", T_max=100),
    "PraNet":            dict(lr=1e-4, scheduler="ReduceLROnPlateau"),  # original PraNet recipe: Adam, lr 1e-4
    "UNet":              dict(lr=1e-4, scheduler="ReduceLROnPlateau"),
    "UNet++":             dict(lr=1e-4, scheduler="ReduceLROnPlateau"),
    "ResUNet":           dict(lr=1e-3, scheduler="ReduceLROnPlateau"),
    "ResUNet++":         dict(lr=1e-3, scheduler="ReduceLROnPlateau"),
    "AttentionUNet":     dict(lr=1e-4, scheduler="ReduceLROnPlateau"),
    "DeepLabV3+":        dict(lr=1e-3, scheduler="ReduceLROnPlateau"),
    "SCA-InceptionUNet": dict(lr=1e-3, scheduler="ReduceLROnPlateau"),
    "TransUNet":         dict(lr=2e-4, scheduler="ReduceLROnPlateau"),
    "UNeXt":             dict(lr=1e-3, scheduler="CosineAnnealingLR", T_max=100),
    "Fast-SCNN":         dict(lr=1e-3, scheduler="CosineAnnealingLR", T_max=100),
    "MobileViT-Seg":     dict(lr=1e-3, scheduler="CosineAnnealingLR", T_max=100),
}


DATASET_PATHS = {
    "CVC-ClinicDB": dict(dataset_path="data/CVC-ClinicDB", image_dir="Original", mask_dir="Ground Truth"),
    "Kvasir-SEG": dict(dataset_path="data/Kvasir-SEG", image_dir="images", mask_dir="masks"),
}


def build_data(dataset: str, batch_size: int, seed: int, group_strategy: str):
    if dataset == "PolypGen":
        ds = PolypGenSegmentationData(
            dataset_path="data/PolypGen2021_segmentation",
            img_size=(256, 256),
            batch_size=batch_size,
            random_state=seed,
        )
    else:
        ds = MedicalImageSegmentationData(
            **DATASET_PATHS[dataset],
            img_size=(256, 256),
            batch_size=batch_size,
            test_size=0.1,
            val_size=0.1,
            random_state=seed,
            binarize_mask=True,
            group_strategy=group_strategy,
        )
    return ds


def run_one(model_name, dataset, seed=0, epochs=100, batch_size=8, group_strategy="phash",
            use_wandb=False, base_dir="results/runs", ds=None, device=None, split_seed=None):
    """Train+evaluate one (model, dataset, seed) combination. Returns the test-results dict.
    Pass a pre-built `ds` (dataset splitter) to reuse the same split across models on one dataset/seed.
    Pass device="cpu" to force CPU (e.g. to avoid contending with a concurrent GPU job)."""
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if ds is None:
        gs = None if group_strategy == "none" else group_strategy
        ss = seed if split_seed is None else split_seed  # split_seed fixed => seeds vary training randomness only
        ds = build_data(dataset, batch_size, ss, gs)
        split_report_path = Path(base_dir) / "split_reports"
        split_report_path.mkdir(parents=True, exist_ok=True)
        with open(split_report_path / f"{dataset}_split{ss}.json", "w") as f:
            json.dump(ds.report.__dict__, f, indent=2, default=str)
    print(ds.report.summary())

    train_loader, val_loader, test_loader = ds.dataloaders()
    model = build_model(model_name, base_channels=12)

    # LiteMedSeg emits (main, aux4, aux3) deep-supervision outputs while
    # training; LightweightLoss is the only criterion that unpacks that tuple.
    criterion = "LightweightLoss" if (model_name in ("LiteMedSeg", "PraNet") or is_ablation(model_name)) else "BCEDiceLoss"

    # ablation variants share LiteMedSeg's recipe (lr 1e-3, cosine, aux-supervised loss)
    hparams = MODEL_HPARAMS["LiteMedSeg" if (is_ablation(model_name) or model_name == "LiteMedSeg-noaux") else model_name]
    cfg = TrainConfig(
        model_name=f"{model_name}_seed{seed}",
        dataset_name=dataset,
        criterion=criterion,
        epochs=epochs,
        base_dir=base_dir,
        use_wandb=use_wandb,
        lr=hparams["lr"],
        scheduler=hparams["scheduler"],
        **({"T_max": hparams["T_max"]} if "T_max" in hparams else {}),
    )

    trainer = SegmentationTrainer(model, device, train_loader, val_loader, test_loader, cfg=cfg)
    train_result = trainer.train()
    print("Training result:", train_result)

    if train_result.get("best_model_path"):
        model.load_state_dict(torch.load(train_result["best_model_path"], map_location=device))
    test_results = trainer.validate_and_save(input_size=(3, 256, 256))
    test_results["model"] = model_name
    test_results["dataset"] = dataset
    test_results["seed"] = seed
    print("Test results:", json.dumps({k: v for k, v in test_results.items()}, indent=2, default=str))
    return test_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, choices=["CVC-ClinicDB", "Kvasir-SEG", "PolypGen"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--group_strategy", default="phash", choices=["phash", "filename_prefix", "none"])
    ap.add_argument("--use_wandb", action="store_true")
    ap.add_argument("--base_dir", default="results/runs")
    ap.add_argument("--split_seed", type=int, default=None,
                    help="Seed for the train/val/test partition. Default: same as --seed (re-split per seed). "
                         "Set to a constant to keep ONE fixed split across training seeds.")
    args = ap.parse_args()
    run_one(args.model, args.dataset, args.seed, args.epochs, args.batch_size,
            args.group_strategy, args.use_wandb, args.base_dir, split_seed=args.split_seed)


if __name__ == "__main__":
    main()
