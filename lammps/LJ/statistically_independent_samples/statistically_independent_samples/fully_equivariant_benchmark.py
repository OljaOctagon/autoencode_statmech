"""Notebook support for the current O(3)-equivariant structural autoencoder.

There are intentionally no top-level training calls here. The benchmark notebook
owns execution, plots, and the 30-seed loop.
"""

from __future__ import annotations

import gc
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class BenchmarkConfig:
    smoke_test: bool = False
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    batch_size: int = 128
    epochs: int = 20
    patience: int = 5
    max_train_per_phase: int = 6_000
    max_validation_per_phase: int = 1_500
    max_test_per_phase: int = 1_500
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    min_noise: float = 0.005
    max_noise: float = 0.08
    grad_clip: float = 1.0

    def __post_init__(self):
        if self.smoke_test:
            self.batch_size = 64
            self.epochs = 2
            self.patience = 2
            self.max_train_per_phase = 256
            self.max_validation_per_phase = 64
            self.max_test_per_phase = 64


@dataclass(frozen=True)
class ModelConfig:
    num_neighbors: int = 18
    hidden_dim: int = 128
    message_dim: int = 128
    num_layers: int = 4
    struct_dim: int = 4
    num_vector_latents: int = 2
    num_tensor_latents: int = 1
    num_rbf: int = 16
    rbf_max_distance: float = 3.0
    decoder_hidden_dim: int = 256
    invariant_decoder_hidden_dim: int = 256
    coordinate_update_scale: float = 0.1


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    epochs: int = 20
    patience: int = 5
    min_noise: float = 0.005
    max_noise: float = 0.08
    lambda_reconstruction: float = 1.0
    lambda_consistency: float = 1.0
    lambda_invariant_geometry: float = 1.0
    lambda_var: float = 1.0
    lambda_cov: float = 0.05
    lambda_rank: float = 0.0
    target_effective_rank: float | None = None
    lambda_scale_cov: float = 0.0
    grad_clip: float = 1.0


@dataclass(frozen=True)
class AblationSpec:
    name: str
    model_config: ModelConfig
    train_config: TrainConfig
    description: str


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, depth=2):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * max(depth - 1, 0) + [out_dim]
        layers = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                layers.append(nn.SiLU())
        self.network = nn.Sequential(*layers)

    def forward(self, values):
        return self.network(values)


class GaussianRBF(nn.Module):
    def __init__(self, count, maximum):
        super().__init__()
        centers = torch.linspace(0.0, maximum, count)
        self.register_buffer("centers", centers)
        spacing = centers[1] - centers[0] if count > 1 else torch.tensor(maximum)
        self.gamma = float(1.0 / (spacing.item() ** 2 + 1e-12))

    def forward(self, distances):
        return torch.exp(-self.gamma * (distances - self.centers) ** 2)


class EGNNBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rbf = GaussianRBF(config.num_rbf, config.rbf_max_distance)
        self.edge_mlp = MLP(
            2 * config.hidden_dim + config.num_rbf,
            config.message_dim,
            config.hidden_dim,
            depth=3,
        )
        self.edge_gate = MLP(config.message_dim, 1, config.hidden_dim)
        self.coordinate_mlp = MLP(config.message_dim, 1, config.hidden_dim)
        self.node_mlp = MLP(
            config.hidden_dim + config.message_dim,
            config.hidden_dim,
            config.hidden_dim,
            depth=3,
        )
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, node_features, coordinates):
        node_count = coordinates.shape[1]
        difference = coordinates[:, :, None, :] - coordinates[:, None, :, :]
        distances = torch.sqrt(difference.square().sum(-1, keepdim=True) + 1e-12)
        source = node_features[:, :, None, :].expand(-1, -1, node_count, -1)
        target = node_features[:, None, :, :].expand(-1, node_count, -1, -1)
        messages = self.edge_mlp(torch.cat([source, target, self.rbf(distances)], -1))
        messages = messages * torch.sigmoid(self.edge_gate(messages))
        non_self = ~torch.eye(node_count, dtype=torch.bool, device=coordinates.device)[None]
        mask = non_self.unsqueeze(-1).to(messages.dtype)
        messages = messages * mask
        degree = non_self.sum(2, keepdim=True).to(messages.dtype)
        aggregated = messages.sum(2) / degree
        node_features = self.norm(
            node_features + self.node_mlp(torch.cat([node_features, aggregated], -1))
        )
        coefficients = (
            self.config.coordinate_update_scale
            * torch.tanh(self.coordinate_mlp(messages))
            * mask
        )
        coordinates = coordinates + (difference * coefficients).sum(2) / degree
        return node_features, coordinates - coordinates[:, :1]


class InvariantPool(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.value = MLP(config.hidden_dim, config.hidden_dim, config.hidden_dim)
        self.logit = MLP(config.hidden_dim, 1, config.hidden_dim)

    def forward(self, node_features):
        return (
            torch.softmax(self.logit(node_features), dim=1)
            * self.value(node_features)
        ).sum(1)


class EquivariantPool(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vector_logits = MLP(
            config.hidden_dim, config.num_vector_latents, config.hidden_dim
        )
        self.tensor_logits = MLP(
            config.hidden_dim, config.num_tensor_latents, config.hidden_dim
        )

    def forward(self, node_features, coordinates):
        vector_weights = torch.softmax(self.vector_logits(node_features), dim=1)
        vectors = torch.einsum("bnk,bnd->bkd", vector_weights, coordinates)
        vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        tensor_weights = torch.softmax(self.tensor_logits(node_features), dim=1)
        outer = coordinates[..., :, None] * coordinates[..., None, :]
        radius_squared = coordinates.square().sum(-1)
        identity = torch.eye(3, dtype=coordinates.dtype, device=coordinates.device)
        traceless = outer - radius_squared[..., None, None] * identity / 3.0
        tensors = torch.einsum("bnk,bnij->bkij", tensor_weights, traceless)
        tensors = tensors / tensors.square().sum(
            (-1, -2), keepdim=True
        ).sqrt().clamp_min(1e-8)
        return {"vector": vectors, "tensor": tensors}


class EquivariantSetDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.basis_count = (
            config.num_vector_latents
            + config.num_vector_latents * config.num_tensor_latents
        )
        self.coefficients = MLP(
            config.struct_dim,
            config.num_neighbors * self.basis_count,
            config.decoder_hidden_dim,
            depth=4,
        )

    def forward(self, structural_latent, equivariant_latent):
        vectors = equivariant_latent["vector"]
        tensor_vectors = torch.einsum(
            "btij,bvj->btvi", equivariant_latent["tensor"], vectors
        ).flatten(1, 2)
        basis = torch.cat([vectors, tensor_vectors], dim=1)
        coefficients = self.coefficients(structural_latent).view(
            len(structural_latent), self.config.num_neighbors, self.basis_count
        )
        return torch.einsum("bna,bad->bnd", coefficients, basis)


class StructuralAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.node_embedding = MLP(2, config.hidden_dim, config.hidden_dim)
        self.blocks = nn.ModuleList([EGNNBlock(config) for _ in range(config.num_layers)])
        self.structural_pool = InvariantPool(config)
        self.structural_head = MLP(
            config.hidden_dim, config.struct_dim, config.hidden_dim, depth=3
        )
        self.equivariant_pool = EquivariantPool(config)
        self.decoder = EquivariantSetDecoder(config)
        signature_dim = config.num_neighbors + config.num_neighbors * (
            config.num_neighbors - 1
        ) // 2
        self.invariant_decoder = MLP(
            config.struct_dim,
            signature_dim,
            config.invariant_decoder_hidden_dim,
            depth=4,
        )

    def encode(self, neighbors):
        normalized, scale = normalize_environment(neighbors)
        center = torch.zeros(
            len(neighbors), 1, 3, dtype=neighbors.dtype, device=neighbors.device
        )
        coordinates = torch.cat([center, normalized], dim=1)
        initial = torch.ones(
            len(neighbors),
            self.config.num_neighbors + 1,
            2,
            dtype=neighbors.dtype,
            device=neighbors.device,
        )
        initial[:, :, 1] = 0.0
        initial[:, 0, 1] = 1.0
        node_features = self.node_embedding(initial)
        for block in self.blocks:
            node_features, coordinates = block(node_features, coordinates)
        node_features, coordinates = node_features[:, 1:], coordinates[:, 1:]
        structural = self.structural_head(self.structural_pool(node_features))
        equivariant = self.equivariant_pool(node_features, coordinates)
        return {
            "z_structure": structural,
            "z_vector": equivariant["vector"],
            "z_tensor": equivariant["tensor"],
            "equivariant": equivariant,
            "scale": scale,
        }

    def forward(self, neighbors):
        output = self.encode(neighbors)
        output["reconstruction"] = self.decoder(
            output["z_structure"], output["equivariant"]
        )
        output["invariant_prediction"] = self.invariant_decoder(output["z_structure"])
        return output


def normalize_environment(coordinates):
    scale = coordinates.square().sum(-1).mean(1).sqrt()[:, None, None].clamp_min(1e-8)
    return coordinates / scale, scale


def invariant_signature(coordinates):
    center_distances = coordinates.norm(dim=-1).sort(1).values
    pairwise = torch.cdist(coordinates, coordinates)
    upper_i, upper_j = torch.triu_indices(
        coordinates.shape[1], coordinates.shape[1], offset=1, device=coordinates.device
    )
    return torch.cat(
        [center_distances, pairwise[:, upper_i, upper_j].sort(1).values], dim=1
    )


def chamfer_per_sample(prediction, target):
    cost = torch.cdist(prediction, target).square()
    return 0.5 * (cost.min(2).values.mean(1) + cost.min(1).values.mean(1))


def random_orthogonal_matrix(batch_size, device, dtype):
    matrix = torch.randn(batch_size, 3, 3, device=device, dtype=dtype)
    orthogonal, triangular = torch.linalg.qr(matrix)
    diagonal = torch.diagonal(triangular, dim1=-2, dim2=-1)
    sign = torch.where(diagonal >= 0, torch.ones_like(diagonal), -torch.ones_like(diagonal))
    return orthogonal * sign[:, None, :]


def rotate(coordinates, rotation):
    return torch.einsum("bij,bnj->bni", rotation, coordinates)


def add_noise(coordinates, sigma):
    if not torch.is_tensor(sigma):
        sigma = coordinates.new_tensor(float(sigma))
    while sigma.ndim < coordinates.ndim:
        sigma = sigma.unsqueeze(-1)
    return coordinates + sigma * torch.randn_like(coordinates)


def effective_rank_tensor(latent):
    centered = latent - latent.mean(0, keepdim=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    return torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())


def total_loss(output_a, output_b, target, config):
    normalized_target, scale = normalize_environment(target)
    signature = invariant_signature(normalized_target)
    reconstruction = 0.5 * (
        chamfer_per_sample(output_a["reconstruction"], normalized_target).mean()
        + chamfer_per_sample(output_b["reconstruction"], normalized_target).mean()
    )
    consistency = F.mse_loss(output_a["z_structure"], output_b["z_structure"])
    invariant = 0.5 * (
        F.mse_loss(output_a["invariant_prediction"], signature)
        + F.mse_loss(output_b["invariant_prediction"], signature)
    )
    latent = torch.cat([output_a["z_structure"], output_b["z_structure"]])
    variance = torch.relu(
        1.0 - torch.sqrt(latent.var(0, unbiased=False) + 1e-4)
    ).mean()
    centered = latent - latent.mean(0, keepdim=True)
    covariance_matrix = centered.T @ centered / max(len(centered) - 1, 1)
    off_diagonal = covariance_matrix - torch.diag(torch.diag(covariance_matrix))
    covariance = off_diagonal.square().sum() / latent.shape[1]
    rank = effective_rank_tensor(latent)
    rank_loss = reconstruction.new_zeros(())
    if config.target_effective_rank is not None:
        rank_loss = torch.relu(
            reconstruction.new_tensor(config.target_effective_rank) - rank
        ).square()
    log_scale = torch.cat([scale.log(), scale.log()]).reshape(-1, 1)
    scale_covariance = (
        centered.T
        @ (log_scale - log_scale.mean(0, keepdim=True))
        / max(len(centered) - 1, 1)
    ).square().sum()
    total = (
        config.lambda_reconstruction * reconstruction
        + config.lambda_consistency * consistency
        + config.lambda_invariant_geometry * invariant
        + config.lambda_var * variance
        + config.lambda_cov * covariance
        + config.lambda_rank * rank_loss
        + config.lambda_scale_cov * scale_covariance
    )
    terms = {
        "total": total,
        "reconstruction": reconstruction,
        "consistency": consistency,
        "invariant_geometry": invariant,
        "variance": variance,
        "covariance": covariance,
        "effective_rank": rank,
        "rank_loss": rank_loss,
        "scale_covariance": scale_covariance,
    }
    return total, {name: value.detach() for name, value in terms.items()}


def make_ablation_specs(config):
    model = ModelConfig(
        hidden_dim=64 if config.smoke_test else 128,
        message_dim=64 if config.smoke_test else 128,
        num_layers=2 if config.smoke_test else 4,
        num_rbf=8 if config.smoke_test else 16,
        decoder_hidden_dim=128 if config.smoke_test else 256,
        invariant_decoder_hidden_dim=128 if config.smoke_test else 256,
    )
    training = TrainConfig(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        epochs=config.epochs,
        patience=config.patience,
        min_noise=config.min_noise,
        max_noise=config.max_noise,
        grad_clip=config.grad_clip,
    )
    return (
        AblationSpec(
            "FULL_A",
            model,
            training,
            "Combined scale-normalized model with consistency",
        ),
        AblationSpec(
            "FULL_B",
            model,
            replace(training, lambda_rank=0.5, target_effective_rank=3.0),
            "FULL_A plus effective-rank regularization",
        ),
        AblationSpec(
            "FULL_C",
            model,
            replace(training, lambda_scale_cov=0.1),
            "FULL_A plus scale-decorrelation regularization",
        ),
    )


def _frame_partition(frame_indices, rng):
    frames = rng.permutation(np.unique(frame_indices))
    if len(frames) < 5:
        raise ValueError("At least five independent frames are required per phase")
    n_test = max(1, round(0.15 * len(frames)))
    n_validation = max(1, round(0.15 * len(frames)))
    partition = np.full(len(frame_indices), "train", dtype="<U5")
    partition[np.isin(frame_indices, frames[:n_test])] = "test"
    partition[np.isin(frame_indices, frames[n_test : n_test + n_validation])] = "val"
    return partition


def _sample(indices, maximum, rng):
    return np.sort(
        indices if len(indices) <= maximum else rng.choice(indices, maximum, replace=False)
    )


def _rms_radius(coordinates):
    return np.sqrt(np.mean(np.sum(np.square(coordinates), axis=-1), axis=-1))


def prepare_dataset(phase_data, phases, label_map, config, seed):
    rng = np.random.default_rng(seed)
    split_names = ("train", "val", "test")
    limits = {
        "train": config.max_train_per_phase,
        "val": config.max_validation_per_phase,
        "test": config.max_test_per_phase,
    }
    storage = {
        split: {"coordinates": [], "labels": [], "scale": []}
        for split in split_names
    }
    plan_rows = []
    for phase in phases:
        data = phase_data[phase]
        coordinates = np.asarray(data["vec_dist"], dtype=np.float32)
        frames = np.asarray(data["frame_indices"])
        if coordinates.shape[1:] != (18, 3):
            raise ValueError(f"{phase}: expected 18x3 coordinates, got {coordinates.shape[1:]}")
        partition = _frame_partition(frames, rng)
        for split in split_names:
            selected = _sample(np.flatnonzero(partition == split), limits[split], rng)
            chosen = coordinates[selected]
            storage[split]["coordinates"].append(chosen)
            storage[split]["labels"].append(
                np.full(len(selected), label_map[phase], dtype=np.int64)
            )
            storage[split]["scale"].append(_rms_radius(chosen))
            plan_rows.append(
                {
                    "phase": phase,
                    "partition": split,
                    "particles": len(selected),
                    "frames": np.unique(frames[selected]).size,
                }
            )
    by_split = {
        split: {
            name: np.concatenate(chunks)
            for name, chunks in storage[split].items()
        }
        for split in split_names
    }
    global_scale = float(np.median(by_split["train"]["scale"]))
    for split in split_names:
        by_split[split]["coordinates"] = (
            by_split[split]["coordinates"] / global_scale
        ).astype(np.float32, copy=False)
    return {
        "by_split": by_split,
        "coordinates": np.concatenate(
            [by_split[split]["coordinates"] for split in split_names]
        ),
        "labels": np.concatenate([by_split[split]["labels"] for split in split_names]),
        "radial_scale": np.concatenate(
            [by_split[split]["scale"] for split in split_names]
        ),
        "split": np.concatenate(
            [
                np.full(len(by_split[split]["labels"]), split, dtype="<U5")
                for split in split_names
            ]
        ),
        "global_scale": global_scale,
        "plan": pd.DataFrame(plan_rows),
    }


def _make_loaders(dataset, config, seed):
    loaders = {}
    for offset, split in enumerate(("train", "val", "test")):
        generator = torch.Generator().manual_seed(seed + offset)
        coordinates = torch.from_numpy(dataset["by_split"][split]["coordinates"])
        loaders[split] = DataLoader(
            TensorDataset(coordinates),
            batch_size=config.batch_size,
            shuffle=split == "train",
            generator=generator if split == "train" else None,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def _run_epoch(model, loader, train_config, benchmark_config, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals, count = {}, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for (clean,) in loader:
            clean = clean.to(benchmark_config.device, non_blocking=True)
            rotation = random_orthogonal_matrix(len(clean), clean.device, clean.dtype)
            target = rotate(clean, rotation)
            sigma = torch.empty(len(clean), device=clean.device).uniform_(
                train_config.min_noise, train_config.max_noise
            )
            view_a, view_b = add_noise(target, sigma), add_noise(target, sigma)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss, terms = total_loss(model(view_a), model(view_b), target, train_config)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_config.grad_clip
                )
                optimizer.step()
            for name, value in terms.items():
                totals[name] = totals.get(name, 0.0) + float(value) * len(clean)
            count += len(clean)
    return {name: value / max(count, 1) for name, value in totals.items()}


def fit_model(dataset, spec, config, seed, verbose):
    set_seed(seed)
    loaders = _make_loaders(dataset, config, seed)
    model = StructuralAutoencoder(spec.model_config).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.train_config.learning_rate,
        weight_decay=spec.train_config.weight_decay,
    )
    history, best_state = [], None
    best_validation, best_epoch, stale = np.inf, -1, 0
    started = time.time()
    for epoch in range(spec.train_config.epochs):
        train_metrics = _run_epoch(
            model, loaders["train"], spec.train_config, config, optimizer
        )
        validation_metrics = _run_epoch(
            model, loaders["val"], spec.train_config, config
        )
        history.append(
            {
                "epoch": epoch + 1,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                },
            }
        )
        score = validation_metrics["total"]
        if verbose:
            print(
                f"{spec.name} seed={seed:>3} epoch={epoch + 1:02d} "
                f"val_total={score:.4f} "
                f"val_recon={validation_metrics['reconstruction']:.4f} "
                f"rank={validation_metrics['effective_rank']:.2f}"
            )
        if score < best_validation - 1e-7:
            best_validation, best_epoch = score, epoch + 1
            best_state, stale = deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= spec.train_config.patience:
                break
    if best_state is None:
        raise RuntimeError("Fully equivariant training produced no checkpoint")
    model.load_state_dict(best_state)
    return {
        "model": model,
        "history": pd.DataFrame(history),
        "best_epoch": best_epoch,
        "best_validation_total": float(best_validation),
        "training_seconds": time.time() - started,
    }


def _collect_outputs(model, coordinates, config):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(coordinates)),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    latent, reconstruction, noise = [], [], []
    model.eval()
    with torch.no_grad():
        for (clean,) in loader:
            clean = clean.to(config.device)
            output = model(clean)
            target, _ = normalize_environment(clean)
            latent.append(output["z_structure"].cpu().numpy())
            reconstruction.append(
                chamfer_per_sample(output["reconstruction"], target).cpu().numpy()
            )
            noisy = model.encode(add_noise(clean, 0.03))["z_structure"]
            noise.append(
                (noisy - output["z_structure"]).square().mean(1).cpu().numpy()
            )
    return {
        "Z": np.concatenate(latent),
        "reconstruction_error": np.concatenate(reconstruction),
        "noise_stability": np.concatenate(noise),
    }


def _effective_rank(latent):
    eigenvalues = np.maximum(
        np.linalg.eigvalsh(np.atleast_2d(np.cov(latent, rowvar=False))), 0.0
    )
    if eigenvalues.sum() <= 1e-12:
        return 0.0
    probabilities = eigenvalues / eigenvalues.sum()
    return float(
        np.exp(
            -np.sum(
                probabilities * np.log(np.clip(probabilities, 1e-12, None))
            )
        )
    )


def _scale_r2(train_z, train_scale, test_z, test_scale):
    coefficients = np.linalg.lstsq(
        np.column_stack([np.ones(len(train_z)), train_z]),
        np.log(train_scale),
        rcond=None,
    )[0]
    target = np.log(test_scale)
    prediction = np.column_stack([np.ones(len(test_z)), test_z]) @ coefficients
    return float(
        1.0
        - np.square(target - prediction).sum()
        / max(np.square(target - target.mean()).sum(), 1e-12)
    )


def _probe(latent, labels, split, seed):
    fit_mask, test_mask = split != "test", split == "test"
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5_000, class_weight="balanced", random_state=seed
        ),
    )
    probe.fit(latent[fit_mask], labels[fit_mask])
    y_test = labels[test_mask]
    y_prediction = probe.predict(latent[test_mask])
    y_score = probe.predict_proba(latent[test_mask])
    return {
        "probe": probe,
        "y_test": y_test,
        "y_pred": y_prediction,
        "accuracy": accuracy_score(y_test, y_prediction),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_prediction),
        "macro_f1": f1_score(y_test, y_prediction, average="macro"),
        "roc_auc": roc_auc_score(
            y_test,
            y_score,
            labels=probe.classes_,
            multi_class="ovr",
            average="macro",
        ),
    }


def _symmetry_audit(model, coordinates, config):
    values = torch.from_numpy(coordinates[:64]).to(config.device)
    model.eval()
    with torch.no_grad():
        rotation = random_orthogonal_matrix(len(values), values.device, values.dtype)
        original, transformed = model(values), model(rotate(values, rotation))
        permutations = torch.stack(
            [
                torch.randperm(values.shape[1], device=values.device)
                for _ in range(len(values))
            ]
        )
        permuted_values = torch.gather(
            values, 1, permutations[..., None].expand(-1, -1, 3)
        )
        permuted = model(permuted_values)
        tensor_target = (
            rotation[:, None]
            @ original["z_tensor"]
            @ rotation[:, None].transpose(-1, -2)
        )
        return {
            "structural_invariance_max_abs": (
                original["z_structure"] - transformed["z_structure"]
            ).abs().max().item(),
            "vector_equivariance_max_abs": (
                torch.einsum("bij,bkj->bki", rotation, original["z_vector"])
                - transformed["z_vector"]
            ).abs().max().item(),
            "tensor_equivariance_max_abs": (
                tensor_target - transformed["z_tensor"]
            ).abs().max().item(),
            "reconstruction_equivariance_max_abs": (
                rotate(original["reconstruction"], rotation)
                - transformed["reconstruction"]
            ).abs().max().item(),
            "permutation_invariance_max_abs": (
                original["z_structure"] - permuted["z_structure"]
            ).abs().max().item(),
        }


def run_suite(
    phase_data: Mapping,
    phases: Sequence[str],
    label_map: Mapping[str, int],
    seed: int,
    config: BenchmarkConfig,
    *,
    keep_artifacts: bool = True,
    verbose: bool = True,
):
    dataset = prepare_dataset(phase_data, phases, label_map, config, seed)
    rows, artifacts = [], {}
    for spec in make_ablation_specs(config):
        if verbose:
            print(f"\nFully equivariant {spec.name} | seed={seed}: {spec.description}")
        fit = fit_model(dataset, spec, config, seed, verbose)
        outputs = _collect_outputs(fit["model"], dataset["coordinates"], config)
        train_mask, test_mask = (
            dataset["split"] == "train",
            dataset["split"] == "test",
        )
        probe = _probe(outputs["Z"], dataset["labels"], dataset["split"], seed)
        metrics = {
            "accuracy": probe["accuracy"],
            "balanced_accuracy": probe["balanced_accuracy"],
            "macro_f1": probe["macro_f1"],
            "roc_auc": probe["roc_auc"],
            "test_reconstruction": float(
                outputs["reconstruction_error"][test_mask].mean()
            ),
            "test_noise_stability": float(outputs["noise_stability"][test_mask].mean()),
            "latent_effective_rank": _effective_rank(outputs["Z"][test_mask]),
            "test_scale_linear_r2": _scale_r2(
                outputs["Z"][train_mask],
                dataset["radial_scale"][train_mask],
                outputs["Z"][test_mask],
                dataset["radial_scale"][test_mask],
            ),
            **_symmetry_audit(fit["model"], dataset["coordinates"][test_mask], config),
        }
        rows.append(
            {
                "seed": seed,
                "variant": spec.name,
                "description": spec.description,
                "n_samples": len(dataset["coordinates"]),
                "coordinate_scale": dataset["global_scale"],
                "best_epoch": fit["best_epoch"],
                "validation_total": fit["best_validation_total"],
                "training_seconds": fit["training_seconds"],
                **metrics,
            }
        )
        if keep_artifacts:
            artifacts[spec.name] = {
                "spec": spec,
                "fit": fit,
                "Z": outputs["Z"],
                "probe": probe,
                "metrics": metrics,
                "labels": dataset["labels"],
                "split": dataset["split"],
                "radial_scale": dataset["radial_scale"],
            }
        del fit, outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(rows), artifacts, dataset


__all__ = ["BenchmarkConfig", "make_ablation_specs", "run_suite"]
