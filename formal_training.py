"""Run the paired AIP304 training sweep using a frozen PlantSeg manifest."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import tempfile
import time

from filelock import FileLock, Timeout
from PIL import Image, ImageOps
import PIL
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF


WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
IMAGE_SIZE = 224
IMAGE_MEAN = WEIGHTS.transforms().mean
IMAGE_STD = WEIGHTS.transforms().std
PAD_COLOR = tuple(round(value * 255) for value in IMAGE_MEAN)
COLOR_JITTER = ColorJitter(brightness=0.1, contrast=0.1)
BATCH_SIZE = 32
EPOCHS = 20
RESUME_INTERVAL = 5
WEIGHT_DECAY = 1e-4
MIN_LEARNING_RATE = 1e-6
LESION_WEIGHT = 0.25
DEFAULT_ASSIGNMENTS = ("A", "B")
DEFAULT_SEEDS = (42, 123, 2026)
DEFAULT_LEARNING_RATES = (1e-4, 3e-4)
MODEL_ARCHITECTURES = {
    "baseline": "resnet18",
    "lesion": "resnet18_with_layer2_lesion_head",
}
CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class RunSpec:
    assignment: str
    seed: int
    learning_rate: float
    model: str

    def __post_init__(self):
        if self.assignment not in DEFAULT_ASSIGNMENTS:
            raise ValueError(f"Unknown assignment: {self.assignment}")
        if self.model not in MODEL_ARCHITECTURES:
            raise ValueError(f"Unknown model variant: {self.model}")
        if not 0 <= self.seed < 2**63:
            raise ValueError("Use a nonnegative integer seed below 2**63.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= MIN_LEARNING_RATE:
            raise ValueError(f"Learning rate must be finite and exceed {MIN_LEARNING_RATE}.")


def resolve_data_root(path=None):
    configured = path or os.environ.get("PLANTSEG_ROOT")
    candidates = (
        [Path(configured).expanduser()]
        if configured
        else [Path.cwd() / "data" / "plantseg", Path.cwd() / "plantseg"]
    )
    for candidate in candidates:
        if (candidate / "Metadata.csv").is_file() and (candidate / "images").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Pass --data-root or set PLANTSEG_ROOT to the extracted PlantSeg directory "
        "containing Metadata.csv and images/."
    )


def load_manifest(path):
    data = Path(path).read_bytes()
    manifest = json.loads(data)
    if manifest.get("release") != "PlantSeg v7":
        raise ValueError("The formal sweep requires a frozen PlantSeg v7 manifest.")
    return manifest, hashlib.sha256(data).hexdigest()


def checkpoint_path(output_dir, spec):
    return Path(output_dir) / (
        f"{spec.model}_{spec.assignment}_seed{spec.seed}_lr{spec.learning_rate:g}_epochs{EPOCHS}.pt"
    )


class ResizedImageCache:
    """Reuse the deterministic resize stage while augmenting every training visit."""

    def __init__(self):
        self.rgb = {}
        self.masks = {}

    def resized_rgb(self, path):
        if path not in self.rgb:
            self.rgb[path] = resize_rgb(path)
        return self.rgb[path]

    def resized_mask(self, path, original_size, resized_size):
        key = (path, original_size, resized_size)
        if key not in self.masks:
            self.masks[key] = resize_mask(path, original_size, resized_size)
        return self.masks[key]


def resize_rgb(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    scale = IMAGE_SIZE / max(image.size)
    size = tuple(max(1, round(value * scale)) for value in image.size)
    return image.resize(size, Image.Resampling.BILINEAR), image.size


def resize_mask(path, original_size, resized_size):
    with Image.open(path) as source:
        if original_size != source.size:
            raise ValueError(f"Mask size {source.size} does not match oriented RGB size {original_size}: {path}")
        mask = source.convert("L").point(lambda value: 255 if value > 0 else 0)
    return mask.resize(resized_size, Image.Resampling.NEAREST)


def prepare_tensor(image, *, training=False, mask=None):
    if training:
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
            if mask is not None:
                mask = ImageOps.mirror(mask)
        image = COLOR_JITTER(image)

    offset = tuple((IMAGE_SIZE - value) // 2 for value in image.size)
    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), PAD_COLOR)
    canvas.paste(image, offset)
    rgb = TF.normalize(TF.to_tensor(canvas), IMAGE_MEAN, IMAGE_STD)
    if mask is None:
        return rgb

    mask_canvas = Image.new("L", canvas.size)
    mask_canvas.paste(mask, offset)
    valid_region = Image.new("L", canvas.size)
    left, top = offset
    valid_region.paste(255, (left, top, left + image.width, top + image.height))
    return rgb, TF.to_tensor(mask_canvas), TF.to_tensor(valid_region)


class RGBDataset(Dataset):
    """Expose RGB and a target index, using -1 for unknown images."""

    def __init__(self, rows, data_root, *, training=False, cache=None):
        self.rows = rows
        self.data_root = Path(data_root)
        self.training = training
        self.cache = cache

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.data_root / row["image_path"]
        image, _ = self.cache.resized_rgb(path) if self.cache else resize_rgb(path)
        rgb = prepare_tensor(image, training=self.training)
        return rgb, -1 if row["target"] is None else row["target"]


class LesionTrainingDataset(RGBDataset):
    """Expose binary lesion masks only for known training rows."""

    def __init__(self, rows, data_root, *, cache=None):
        if any(row["split"] != "train" or row["role"] != "known" for row in rows):
            raise ValueError("Lesion supervision requires known-class training rows.")
        super().__init__(rows, data_root, training=True, cache=cache)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.data_root / row["image_path"]
        mask_path = self.data_root / row["mask_path"]
        if self.cache:
            image, original_size = self.cache.resized_rgb(path)
            mask = self.cache.resized_mask(mask_path, original_size, image.size)
        else:
            image, original_size = resize_rgb(path)
            mask = resize_mask(mask_path, original_size, image.size)
        rgb, mask_tensor, valid_tensor = prepare_tensor(image, training=True, mask=mask)
        return rgb, row["target"], mask_tensor, valid_tensor


class LesionSupervisedResNet18(nn.Module):
    def __init__(self, num_classes, weights=None):
        super().__init__()
        self.classifier = resnet18(weights=weights)
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)
        # Preserve the CPU augmentation RNG stream used by the paired baseline.
        with torch.random.fork_rng(devices=[]):
            self.lesion_head = nn.Sequential(
                nn.Conv2d(128, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 1, kernel_size=1),
            )

    def forward(self, images, *, predict_lesions=False):
        if not predict_lesions:
            return self.classifier(images)

        backbone = self.classifier
        features = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(images))))
        lesion_features = backbone.layer2(backbone.layer1(features))
        features = backbone.layer4(backbone.layer3(lesion_features))
        logits = backbone.fc(torch.flatten(backbone.avgpool(features), 1))
        lesion_logits = F.interpolate(
            self.lesion_head(lesion_features),
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits, lesion_logits


def build_model(num_classes, variant, pretrained=False):
    weights = WEIGHTS if pretrained else None
    if variant == "lesion":
        return LesionSupervisedResNet18(num_classes, weights=weights)
    if variant != "baseline":
        raise ValueError(f"Unknown model variant: {variant}")
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def lesion_loss(logits, target, valid_region):
    """Average per-image BCE and soft Dice over the unpadded area."""
    axes = (1, 2, 3)
    pixel_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (pixel_loss * valid_region).sum(axes) / valid_region.sum(axes).clamp_min(1)
    probability = logits.sigmoid() * valid_region
    target = target * valid_region
    intersection = (probability * target).sum(axes)
    dice = (2 * intersection + 1) / (probability.sum(axes) + target.sum(axes) + 1)
    return 0.5 * (bce.mean() + (1 - dice).mean())


def resolve_device(device=None):
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def paired_rows(manifest, assignment):
    class_order = manifest["known_class_order"][assignment]
    rows = manifest["samples"][assignment]
    train = [row for row in rows if row["split"] == "train"]
    val = [row for row in rows if row["split"] == "val" and row["role"] == "known"]
    if len(class_order) != 25 or len(set(class_order)) != 25 or not train or not val:
        raise ValueError(f"Assignment {assignment} must contain 25 known classes and train/val rows.")
    for row in train + val:
        target = row["target"]
        if row["role"] != "known" or not isinstance(target, int) or not 0 <= target < len(class_order):
            raise ValueError(f"Invalid known train/val row: {row}")
        if class_order[target] != row["class_name"]:
            raise ValueError(f"Target and known class order disagree: {row}")
    if {row["target"] for row in train} != set(range(len(class_order))):
        raise ValueError(f"Assignment {assignment} is missing training examples for a known class.")
    return train, val, class_order


def train_epoch(model, loader, optimizer, criterion, device, use_lesions):
    model.train()
    classification_losses, auxiliary_losses = [], []
    for images, targets, *auxiliary_targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        if use_lesions:
            masks, valid = (value.to(device) for value in auxiliary_targets)
            logits, mask_logits = model(images, predict_lesions=True)
            classification = criterion(logits, targets)
            auxiliary = lesion_loss(mask_logits, masks, valid)
            loss = classification + LESION_WEIGHT * auxiliary
            auxiliary_losses.append(auxiliary.detach() * targets.numel())
        else:
            classification = criterion(model(images), targets)
            loss = classification
        loss.backward()
        optimizer.step()
        classification_losses.append(classification.detach() * targets.numel())

    classification_mean = torch.stack(classification_losses).sum().item() / len(loader.dataset)
    auxiliary_mean = (
        torch.stack(auxiliary_losses).sum().item() / len(loader.dataset)
        if use_lesions else None
    )
    return classification_mean, auxiliary_mean


@torch.inference_mode()
def validate_known(model, loader, criterion, device, num_classes):
    model.eval()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    loss_sum = 0.0
    for images, targets in loader:
        logits = model(images.to(device))
        loss_sum += criterion(logits, targets.to(device)).item() * targets.numel()
        predicted = logits.argmax(dim=1).cpu()
        pairs = targets * num_classes + predicted
        confusion += torch.bincount(pairs, minlength=num_classes**2).reshape(confusion.shape)
    true_positive = confusion.diag()
    denominator = confusion.sum(dim=0) + confusion.sum(dim=1)
    macro_f1 = (2 * true_positive / denominator.clamp_min(1)).mean().item()
    return loss_sum / len(loader.dataset), macro_f1, true_positive.sum().item() / len(loader.dataset)


def checkpoint_settings(spec, manifest_sha256, class_order):
    settings = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": MODEL_ARCHITECTURES[spec.model],
        "pretrained_weights": WEIGHTS.name,
        "assignment": spec.assignment,
        "seed": spec.seed,
        "known_class_order": list(class_order),
        "manifest_sha256": manifest_sha256,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "initial_learning_rate": spec.learning_rate,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR",
        "minimum_learning_rate": MIN_LEARNING_RATE,
        "epochs": EPOCHS,
        "image_size": IMAGE_SIZE,
    }
    if spec.model == "lesion":
        settings["lesion_loss_weight"] = LESION_WEIGHT
    return settings


def check_settings(path, checkpoint, settings):
    mismatched = [key for key, expected in settings.items() if checkpoint.get(key) != expected]
    if mismatched:
        raise ValueError(f"{path} has mismatched settings: {', '.join(mismatched)}")


def read_checkpoint(path, spec, manifest_sha256, class_order):
    """Load a completed formal run after checking its provenance and selection."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    check_settings(path, checkpoint, checkpoint_settings(spec, manifest_sha256, class_order))
    history = checkpoint["history"]
    if [row["epoch"] for row in history] != list(range(1, EPOCHS + 1)):
        raise ValueError(f"Expected exactly {EPOCHS} completed epochs in {path}.")
    if any(not math.isfinite(row["val_macro_f1"]) for row in history):
        raise ValueError(f"Nonfinite validation macro-F1 in {path}.")
    best = max(history, key=lambda row: row["val_macro_f1"])
    if checkpoint.get("selected_epoch") != best["epoch"] or not checkpoint.get("model_state_dict"):
        raise ValueError(f"Invalid selected epoch or model state in {path}.")
    return checkpoint


def save_atomically(data, path):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(data, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def cpu_model_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def runtime_info(device):
    info = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "pillow": PIL.__version__,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "device": str(device),
        "torch_cpu_threads": torch.get_num_threads(),
    }
    if device.type == "cuda":
        info["cuda_device"] = torch.cuda.get_device_name(device)
    return info


def capture_rng(generator, device):
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "training_loader": generator.get_state(),
    }
    if device.type == "cuda":
        state["accelerator"] = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        state["accelerator"] = torch.mps.get_rng_state()
    return state


def restore_rng(state, generator, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["training_loader"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["accelerator"], device)
    elif device.type == "mps":
        torch.mps.set_rng_state(state["accelerator"])


def train_run(spec, manifest_path, data_root, output_dir, device=None, *, cache=None):
    """Train or resume a run using only known train/validation samples."""
    destination = checkpoint_path(output_dir, spec)
    lock_dir = destination.parent / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_dir / (destination.name + ".lock")), timeout=0)
    try:
        lock.acquire()
    except Timeout as error:
        raise RuntimeError(f"Another process is running {destination.name}.") from error
    try:
        return _train_run_locked(spec, manifest_path, data_root, destination, device, cache)
    finally:
        lock.release()


def _train_run_locked(spec, manifest_path, data_root, destination, device, cache):
    manifest, manifest_sha256 = load_manifest(manifest_path)
    train_rows, val_rows, class_order = paired_rows(manifest, spec.assignment)
    partial = destination.with_suffix(".resume.pt")
    if destination.is_file():
        read_checkpoint(destination, spec, manifest_sha256, class_order)
        partial.unlink(missing_ok=True)
        print(f"Completed checkpoint already verified: {destination.name}", flush=True)
        return destination

    device = resolve_device(device)
    settings = checkpoint_settings(spec, manifest_sha256, class_order)
    runtime = runtime_info(device)
    use_lesions = spec.model == "lesion"
    cache = cache if cache is not None else ResizedImageCache()
    data_root = resolve_data_root(data_root)

    random.seed(spec.seed)
    torch.manual_seed(spec.seed)
    generator = torch.Generator().manual_seed(spec.seed)
    train_dataset = (
        LesionTrainingDataset(train_rows, data_root, cache=cache)
        if use_lesions else RGBDataset(train_rows, data_root, training=True, cache=cache)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(RGBDataset(val_rows, data_root, cache=cache), batch_size=BATCH_SIZE, num_workers=0)
    n_classes = len(class_order)
    model = build_model(n_classes, spec.model, pretrained=not partial.exists()).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LEARNING_RATE)
    history = []
    best_state = None
    best_result = None
    elapsed_before = 0.0
    started_at = datetime.now(timezone.utc).isoformat()
    if partial.is_file():
        resume = torch.load(partial, map_location="cpu", weights_only=True)
        check_settings(partial, resume, {**settings, "runtime": runtime})
        history = resume["history"]
        if [row["epoch"] for row in history] != list(range(1, len(history) + 1)) or not 0 < len(history) < EPOCHS:
            raise ValueError(f"Invalid progress history in {partial}.")
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        restore_rng(resume["rng"], generator, device)
        best_result = max(history, key=lambda row: row["val_macro_f1"])
        best_state = resume["best_model_state_dict"]
        elapsed_before = resume["elapsed_seconds"]
        started_at = resume["started_at_utc"]
        del resume

    print(
        f"Starting {destination.name} on {device} with {len(train_rows)} train and "
        f"{len(val_rows)} known validation images at epoch {len(history) + 1}/{EPOCHS}.",
        flush=True,
    )
    start = time.perf_counter()
    for epoch in range(len(history) + 1, EPOCHS + 1):
        epoch_start = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss, aux_loss = train_epoch(model, train_loader, optimizer, criterion, device, use_lesions)
        val_loss, val_f1, val_accuracy = validate_known(model, val_loader, criterion, device, n_classes)
        scheduler.step()
        if not all(math.isfinite(value) for value in (train_loss, val_loss, val_f1, val_accuracy)):
            raise FloatingPointError(f"Nonfinite loss or metrics in {destination.name}, epoch {epoch}.")
        result = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_classification_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
            "val_accuracy": val_accuracy,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        if use_lesions:
            if not math.isfinite(aux_loss):
                raise FloatingPointError(f"Nonfinite lesion loss in {destination.name}, epoch {epoch}.")
            result["train_lesion_loss"] = aux_loss
        else:
            result["train_loss"] = train_loss
        history.append(result)
        if best_result is None or val_f1 > best_result["val_macro_f1"]:
            best_result = result
            best_state = cpu_model_state(model)

        if epoch % RESUME_INTERVAL == 0 and epoch < EPOCHS:
            save_atomically(
                {
                    **settings,
                    "runtime": runtime,
                    "history": history,
                    "model_state_dict": cpu_model_state(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "rng": capture_rng(generator, device),
                    "best_model_state_dict": best_state,
                    "elapsed_seconds": elapsed_before + time.perf_counter() - start,
                    "started_at_utc": started_at,
                },
                partial,
            )

        print(
            f"{spec.model} {spec.assignment} seed {spec.seed} lr {spec.learning_rate:g} "
            f"epoch {epoch:02d}/{EPOCHS} | val macro-F1 {val_f1:.4f} | "
            f"val accuracy {val_accuracy:.2%} | {result['epoch_seconds']:.1f}s",
            flush=True,
        )

    checkpoint = {
        **settings,
        "runtime": runtime,
        "train_count": len(train_rows),
        "known_validation_count": len(val_rows),
        "history": history,
        "selected_epoch": best_result["epoch"],
        "elapsed_seconds": elapsed_before + time.perf_counter() - start,
        "model_state_dict": best_state,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_atomically(checkpoint, destination)
    partial.unlink(missing_ok=True)
    print(
        f"Saved {destination.name} at epoch {best_result['epoch']} "
        f"(macro-F1 {best_result['val_macro_f1']:.4f}).",
        flush=True,
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assignments", nargs="+", choices=DEFAULT_ASSIGNMENTS, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--learning-rates", nargs="+", type=float, default=DEFAULT_LEARNING_RATES)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_ARCHITECTURES), default=tuple(MODEL_ARCHITECTURES))
    parser.add_argument("--device", help="PyTorch device such as cpu, mps or cuda. Defaults to automatic selection.")
    args = parser.parse_args()
    data_root = resolve_data_root(args.data_root)
    device = resolve_device(args.device)
    cache = ResizedImageCache()
    configs = list(itertools.product(args.assignments, args.seeds, args.learning_rates, args.models))
    for index, values in enumerate(configs, 1):
        spec = RunSpec(*values)
        print(f"Run {index}/{len(configs)}: {checkpoint_path(args.output_dir, spec).name}", flush=True)
        train_run(spec, args.manifest, data_root, args.output_dir, device, cache=cache)


if __name__ == "__main__":
    main()
