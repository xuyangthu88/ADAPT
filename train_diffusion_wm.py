import argparse
import copy
import glob
import json
import logging
import math
import os
import random
import re
import sys
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Avoid PyTorch backend memory-format mismatches in the original TemporalUnet
# (Conv1d/GroupNorm/Mish) without changing the repository model code.
if hasattr(torch.backends, "mkldnn"):
    torch.backends.mkldnn.enabled = False
if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

import diffuser.models as models
import diffuser.utils as utils
from diffuser.utils.training import reparameterize


Batch = namedtuple("Batch", "trajectories conditions")


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

PROJ_ROOT = Path(os.environ.get("DIFFUSER_WM_PROJ_ROOT", Path(__file__).resolve().parent))
DEFAULT_DATASET = Path(
    os.environ.get(
        "DIFFUSER_WM_DATASET",
        "/home/users/create/smrksun/world_model/occmamba/Aggregated_Cleaned2/Dataset_full/Xy_seq_flat_selected_global.csv",
    )
)
BASE_DIR = (PROJ_ROOT / "results" / "diffusion_wm").resolve()
FULL_RESULTS_DIR = (PROJ_ROOT / "full_results" / "diffusion_wm" / "Full_Results").resolve()


def make_out_dirs(model_name: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = (BASE_DIR / f"{model_name}_Results_{timestamp}").resolve()
    pred_dir = out_dir / "preds"
    ckpt_dir = out_dir / "checkpoints"
    plot_dir = out_dir / "plots"
    log_dir = out_dir / "logs"
    for d in (out_dir, pred_dir, ckpt_dir, plot_dir, log_dir, FULL_RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return out_dir, pred_dir, ckpt_dir, plot_dir, log_dir


# -----------------------------------------------------------------------------
# Defaults aligned with train_mambaformer_wm.py unless diffusion-specific.
# -----------------------------------------------------------------------------

SEED = 3407
TEST_SIZE = 0.2
BATCH_SIZE = 64
EPOCHS = 200
N_STEPS_PER_EPOCH = 5000 #default 10000
LR = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_PATIENCE = 12
MIN_DELTA_RMSE = 5e-5
SAVE_INTERVAL = 30
LOG_INTERVAL = 5
CVRMSE_ON_ORIGINAL = True

HORIZON = 1
WM_HORIZON_MODE = "flat"
N_DIFFUSION_STEPS = 20
PREDICT_EPSILON = False
LOSS_TYPE = "l1" #"smoothl1"
SMOOTHL1_BETA = 0.05
ACTION_WEIGHT = 10
LOSS_DISCOUNT = 1.0
TARGET_LOSS_WEIGHTS = ""
RC_LOSS_WEIGHT = 0.2
RC_DT = 1.0
RC_HUBER_BETA = 0.25
RC_TRUE_WEIGHT = 0.5
RC_BIAS_WEIGHT = 1e-1
RC_RETURN_DRIVER = "pred"  #pred           # 默认，return_temp 参与 RC 梯度
                    #true           # return driver 用真实值
                    #pred_detached  # 用预测 return，但不传梯度
DIM = 32
DIM_MULTS = (8,)
ATTENTION = False
EMA_DECAY = 0.995
GRADIENT_ACCUMULATE_EVERY = 2
UPDATE_EMA_EVERY = 10
Z_DIM = 1
USE_LATENT_ENCODER = False
CONSTANT_Z_VALUE = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTDOOR_MIN = 0.0
OUTDOOR_MAX = 40.0
SEMANTIC_MIN = {
    "room_temp": 18.0,
    "fan": 0.0,
    "supply_temp": 5.0,
    "return_temp": 5.0,
    "occupant_num": 0.0,
}
SEMANTIC_MAX = {
    "room_temp": 30.0,
    "fan": 3.0,
    "supply_temp": 10.0,
    "return_temp": 10.0,
    "occupant_num": 5.0,
}

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Dataset adapter for the original diffuser Trainer
# -----------------------------------------------------------------------------

class IdentityNormalizer:
    """Minimal normalizer object needed by diffuser.utils.Trainer."""

    def __init__(self, action_dim: int, observation_dim: int):
        self.means = {
            "actions": np.zeros(action_dim, dtype=np.float32),
            "observations": np.zeros(observation_dim, dtype=np.float32),
        }
        self.stds = {
            "actions": np.ones(action_dim, dtype=np.float32),
            "observations": np.ones(observation_dim, dtype=np.float32),
        }


class WindowedWorldModelDataset(torch.utils.data.Dataset):
    """
    Presents the windowed CSV as the Batch format consumed by utils.Trainer.

    In flat mode, the full future N-step target vector is one action at one
    diffusion timestep. In temporal mode, the target is reshaped to [N, F] so
    the diffusion horizon matches the forecast horizon.
    """

    def __init__(self, conditions: np.ndarray, targets: np.ndarray, horizon: int = 1):
        self.conditions = np.asarray(conditions, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.horizon = int(horizon)
        if self.horizon < 1:
            raise ValueError("Dataset horizon must be >= 1.")
        if self.targets.shape[1] % self.horizon != 0:
            raise ValueError(
                f"Target dim {self.targets.shape[1]} is not divisible by horizon {self.horizon}."
            )
        self.observation_dim = self.conditions.shape[1]
        self.flat_target_dim = self.targets.shape[1]
        self.action_dim = self.targets.shape[1] // self.horizon
        self.normalizer = IdentityNormalizer(self.action_dim, self.observation_dim)

    def __len__(self):
        return self.conditions.shape[0]

    def __getitem__(self, idx):
        obs = np.ascontiguousarray(self.conditions[idx], dtype=np.float32)
        act = np.ascontiguousarray(
            self.targets[idx].reshape(self.horizon, self.action_dim),
            dtype=np.float32,
        )
        obs_seq = np.repeat(obs[None, :], self.horizon, axis=0)
        trajectory = np.ascontiguousarray(np.concatenate([act, obs_seq], axis=-1), dtype=np.float32)
        return Batch(trajectory, {t: obs for t in range(self.horizon)})


class NullRenderer:
    pass


# -----------------------------------------------------------------------------
# Column construction and normalization, matched to train_mambaformer_wm.py
# -----------------------------------------------------------------------------

def try_load_manifest(csv_path: Path):
    manifest_paths = [
        csv_path.with_name("manifest_global.json"),
        PROJ_ROOT / "Aggregated_Cleaned" / "Dataset_full" / "manifest_global.json",
        PROJ_ROOT / "Aggregated_Cleaned" / "manifest_global.json",
    ]
    for manifest_path in manifest_paths:
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            T = int(data.get("T", 0)) or None
            N = int(data.get("N", 0)) or None
            feature_bases = data.get("feature_bases")
            target_bases = data.get("target_bases")
            if T and N and feature_bases and target_bases:
                return T, N, list(feature_bases), list(target_bases), manifest_path
    return None, None, None, None, None


def infer_T(columns):
    values = []
    for c in columns:
        m = re.search(r"_t(\d+)$", str(c))
        if m:
            values.append(int(m.group(1)))
    return max(values) if values else 1


def infer_N(columns):
    values = []
    for c in columns:
        if str(c).startswith("y_"):
            m = re.search(r"_t(\d+)$", str(c))
            if m:
                values.append(int(m.group(1)))
    return max(values) if values else 1


def split_by_t(columns, T):
    by_t = {t: set() for t in range(1, T + 1)}
    map_tf = {}
    for c in columns:
        m = re.search(r"(.*)_t(\d+)$", str(c))
        if not m:
            continue
        base, t = m.group(1), int(m.group(2))
        if 1 <= t <= T:
            by_t[t].add(base)
            map_tf[(t, base)] = c
    return by_t, map_tf


def build_order_fallback(df_cols, T):
    feats = [c for c in df_cols if not str(c).startswith("y_") and c not in ("episode", "t")]
    by_t, map_tf = split_by_t(feats, T)
    if not by_t or any(len(s) == 0 for s in by_t.values()):
        return [], 0
    common = sorted(set.intersection(*by_t.values()))
    ordered = []
    for t in range(1, T + 1):
        for b in common:
            ordered.append(map_tf[(t, b)])
    return ordered, len(common)


def build_columns(df: pd.DataFrame, csv_path: Path):
    T_m, N_m, feature_bases, target_bases, manifest_path = try_load_manifest(csv_path)
    if T_m and N_m and feature_bases and target_bases:
        T, N = T_m, N_m
        input_bases = feature_bases
    else:
        feature_cols = [c for c in df.columns if not str(c).startswith("y_") and c not in ("episode", "t")]
        target_cols_raw = [c for c in df.columns if str(c).startswith("y_")]
        T, N = infer_T(feature_cols), infer_N(target_cols_raw)
        ordered_cols, F_common = build_order_fallback(df.columns, T)
        input_bases = [re.sub(r"_t\d+$", "", c) for c in ordered_cols[:F_common]]
        target_bases = sorted(
            {
                re.sub(r"_t\d+$", "", c.replace("y_", "", 1))
                for c in target_cols_raw
            }
        )
        manifest_path = None

    input_cols = []
    for t in range(1, T + 1):
        for base in input_bases:
            col = f"{base}_t{t}"
            if col in df.columns:
                input_cols.append(col)

    target_cols = []
    for k in range(1, N + 1):
        for base in target_bases:
            col = f"y_{base}_t{k}"
            if col in df.columns:
                target_cols.append(col)

    if len(input_cols) != T * len(input_bases):
        raise ValueError("Input columns are not a complete context grid.")
    if not target_cols or len(target_cols) % N != 0:
        raise ValueError("Target columns are missing or not divisible by forecast steps.")
    return T, N, input_bases, target_bases, input_cols, target_cols, manifest_path


def _base_range(base: str):
    if base == "outdoor_temp":
        return OUTDOOR_MIN, OUTDOOR_MAX
    if base == "FCU_fan_feedback":
        return SEMANTIC_MIN["fan"], SEMANTIC_MAX["fan"]
    if base.startswith("occupant_") or base.startswith("occupant_num"):
        return SEMANTIC_MIN["occupant_num"], SEMANTIC_MAX["occupant_num"]
    if base in SEMANTIC_MIN:
        return SEMANTIC_MIN[base], SEMANTIC_MAX[base]
    raise KeyError(f"Unknown base for normalization: {base}")


def _semantic_base(col: str):
    if col.startswith("y_"):
        base = re.sub(r"_t\d+$", "", col.replace("y_", "", 1))
    else:
        base = re.sub(r"_t\d+$", "", col)
    return re.sub(r"^room\d+_", "", base)


def _mins_maxs_for_columns(columns: List[str]):
    mins, maxs = [], []
    for col in columns:
        lo, hi = _base_range(_semantic_base(col))
        mins.append(lo)
        maxs.append(hi)
    return np.asarray(mins, dtype=np.float32), np.asarray(maxs, dtype=np.float32)


def _normalize_matrix(arr: np.ndarray, columns: List[str]) -> np.ndarray:
    mins, maxs = _mins_maxs_for_columns(columns)
    denom = np.clip(maxs - mins, 1e-6, None)
    return (arr - mins) / denom


def _denormalize_matrix(arr: np.ndarray, columns: List[str]) -> np.ndarray:
    mins, maxs = _mins_maxs_for_columns(columns)
    denom = np.clip(maxs - mins, 1e-6, None)
    return arr * denom + mins


class MultiZoneRCLoss(torch.nn.Module):
    """
    Stable whole-building RC residual for predicted future room temperatures.

    The dynamics are
        C dT/dt = -K T + k_out T_out - b_hvac phi(fan) (T_room - T_supply) + c_occ O + d
    with K = L + diag(k_out). L is a learned symmetric graph Laplacian, so the
    internal room-to-room exchange is dissipative without needing a floor graph.
    phi(fan) is a learned monotone nonlinear map for the discrete FCU fan modes.
    """

    def __init__(
        self,
        schema,
        target_mins,
        target_maxs,
        condition_mins,
        condition_maxs,
        weight=0.05,
        dt=1.0,
        huber_beta=0.25,
        true_weight=1.0,
        bias_weight=1e-4,
        return_driver="pred",
        eps=1e-3,
    ):
        super().__init__()
        self.pred_weight = float(weight)
        self.dt = float(dt)
        self.huber_beta = float(huber_beta)
        self.true_weight = float(true_weight)
        self.bias_weight = float(bias_weight)
        self.return_driver = str(return_driver).lower()
        if self.return_driver not in ("pred", "true", "pred_detached"):
            raise ValueError("return_driver must be one of: pred, true, pred_detached.")
        self.eps = float(eps)
        self.num_rooms = int(schema["num_rooms"])
        self.forecast_steps = int(schema["forecast_steps"])
        self.condition_dim = int(schema["condition_dim"])

        self.register_buffer("target_mins", torch.as_tensor(target_mins, dtype=torch.float32))
        self.register_buffer("target_ranges", torch.as_tensor(target_maxs - target_mins, dtype=torch.float32))
        self.register_buffer("condition_mins", torch.as_tensor(condition_mins, dtype=torch.float32))
        self.register_buffer("condition_ranges", torch.as_tensor(condition_maxs - condition_mins, dtype=torch.float32))

        for key in (
            "target_room_temp_idx",
            "target_outdoor_idx",
            "target_occupancy_idx",
            "target_occupancy_mask",
            "target_supply_idx",
            "current_room_temp_idx",
            "current_occupancy_idx",
            "current_occupancy_mask",
            "current_fan_idx",
            "current_supply_idx",
        ):
            self.register_buffer(key, torch.as_tensor(schema[key], dtype=torch.long))
        self.register_buffer("current_outdoor_idx", torch.as_tensor(schema["current_outdoor_idx"], dtype=torch.long))

        self.raw_capacity = torch.nn.Parameter(torch.full((self.num_rooms,), 1.0))
        self.raw_outdoor = torch.nn.Parameter(torch.full((self.num_rooms,), -3.0))
        self.raw_internal = torch.nn.Parameter(torch.full((self.num_rooms, self.num_rooms), -4.0))
        self.raw_hvac = torch.nn.Parameter(torch.full((self.num_rooms,), -3.0))
        fan_increment_init = math.log(math.expm1(1.0))
        self.raw_fan_mode_increments = torch.nn.Parameter(
            torch.full((self.num_rooms, 3), fan_increment_init)
        )
        self.raw_occupancy = torch.nn.Parameter(torch.full((self.num_rooms,), -3.0))
        self.bias = torch.nn.Parameter(torch.zeros(self.num_rooms))

    def _denormalize_targets(self, target_norm):
        return target_norm * self.target_ranges + self.target_mins

    def _denormalize_conditions(self, condition_norm):
        condition_norm = condition_norm[:, : self.condition_dim]
        return condition_norm * self.condition_ranges + self.condition_mins

    def _gather(self, values, indices):
        safe_indices = indices.clamp_min(0)
        return values[:, safe_indices]

    def _sum_occupancy(self, values, indices, mask):
        occ = self._gather(values, indices)
        return (occ * mask.to(dtype=values.dtype)).sum(dim=-1)

    def _stable_k(self, device, dtype):
        eye = torch.eye(self.num_rooms, device=device, dtype=dtype)
        g = torch.nn.functional.softplus(0.5 * (self.raw_internal + self.raw_internal.T))
        g = g * (1.0 - eye)
        laplacian = torch.diag(g.sum(dim=1)) - g
        k_out = torch.nn.functional.softplus(self.raw_outdoor) + self.eps
        k = laplacian + torch.diag(k_out)
        return k, k_out

    def _fan_mode_coefficients(self, device, dtype):
        increments = torch.nn.functional.softplus(self.raw_fan_mode_increments).to(device=device, dtype=dtype)
        cumulative = torch.cumsum(increments, dim=-1)
        active = 3.0 * cumulative / (cumulative[:, -1:] + self.eps)
        off = torch.zeros((self.num_rooms, 1), device=device, dtype=dtype)
        return torch.cat([off, active], dim=-1)

    def _fan_mode_gain(self, fan_mode):
        coeffs = self._fan_mode_coefficients(fan_mode.device, fan_mode.dtype)
        mode = fan_mode.round().to(dtype=torch.long).clamp(0, coeffs.shape[-1] - 1)
        coeffs = coeffs.view(*([1] * (mode.dim() - 1)), self.num_rooms, coeffs.shape[-1])
        coeffs = coeffs.expand(*mode.shape, coeffs.shape[-1])
        return torch.gather(coeffs, dim=-1, index=mode.unsqueeze(-1)).squeeze(-1)

    def _flatten_target_prediction(self, x):
        if x.shape[1] == self.forecast_steps:
            step_dim = self.target_mins.numel() // self.forecast_steps
            return x[:, :, :step_dim].reshape(x.shape[0], self.target_mins.numel())
        return x[:, 0, : self.target_mins.numel()]

    def _rc_residual(self, t_seq, out_driver, occ_driver, hvac_driver):
        t_now = t_seq[:, :-1, :]
        d_t = (t_seq[:, 1:, :] - t_now) / self.dt

        k, k_out = self._stable_k(t_now.device, t_now.dtype)
        capacity = torch.nn.functional.softplus(self.raw_capacity).to(dtype=t_now.dtype) + self.eps
        hvac_gain = torch.nn.functional.softplus(self.raw_hvac).to(dtype=t_now.dtype)
        occupancy_gain = torch.nn.functional.softplus(self.raw_occupancy).to(dtype=t_now.dtype)
        bias = self.bias.to(dtype=t_now.dtype)

        rhs = -torch.einsum("ij,bkj->bki", k, t_now)
        rhs = rhs + out_driver.unsqueeze(-1) * k_out.to(dtype=t_now.dtype)
        rhs = rhs + hvac_driver * hvac_gain
        rhs = rhs + occ_driver * occupancy_gain + bias
        rc_rate = rhs / capacity
        return d_t - rc_rate

    def _huber_zero(self, residual):
        return torch.nn.functional.smooth_l1_loss(
            residual,
            torch.zeros_like(residual),
            beta=self.huber_beta,
            reduction="mean",
        )

    def forward(self, x0, cond, x_start=None):
        pred_norm = self._flatten_target_prediction(x0)
        pred = self._denormalize_targets(pred_norm)
        true = self._denormalize_targets(self._flatten_target_prediction(x_start)) if x_start is not None else pred.detach()
        condition = self._denormalize_conditions(cond[0])

        pred_t = pred[:, self.target_room_temp_idx]
        true_t = true[:, self.target_room_temp_idx].detach()

        true_out = true[:, self.target_outdoor_idx].detach()
        true_occ = self._sum_occupancy(true, self.target_occupancy_idx, self.target_occupancy_mask).detach()
        true_supply = true[:, self.target_supply_idx].detach()

        current_t = condition[:, self.current_room_temp_idx]
        current_out = condition[:, self.current_outdoor_idx].detach()
        current_occ = self._sum_occupancy(
            condition, self.current_occupancy_idx, self.current_occupancy_mask
        ).detach()
        current_fan = condition[:, self.current_fan_idx].detach()
        current_fan_gain = self._fan_mode_gain(current_fan)
        # Cooling season: supply water is colder than room air, so this is a negative cooling flux.
        current_supply = condition[:, self.current_supply_idx].detach()
        current_hvac = -current_fan_gain * (current_t.detach() - current_supply)

        t_seq = torch.cat([current_t.unsqueeze(1), pred_t], dim=1)
        true_t_seq = torch.cat([current_t.detach().unsqueeze(1), true_t], dim=1)
        out_driver = torch.cat([current_out[:, None], true_out[:, :-1]], dim=1)
        occ_driver = torch.cat([current_occ.unsqueeze(1), true_occ[:, :-1, :]], dim=1)
        held_fan_gain = current_fan_gain.unsqueeze(1).expand(-1, self.forecast_steps, -1)
        pred_hvac_future = -held_fan_gain[:, 1:, :] * (pred_t[:, :-1, :] - true_supply[:, :-1, :])
        true_hvac_future = -held_fan_gain[:, 1:, :] * (true_t[:, :-1, :] - true_supply[:, :-1, :])
        pred_hvac_driver = torch.cat([current_hvac.unsqueeze(1), pred_hvac_future], dim=1)
        true_hvac_driver = torch.cat([current_hvac.unsqueeze(1), true_hvac_future], dim=1)

        pred_residual = self._rc_residual(t_seq, out_driver, occ_driver, pred_hvac_driver)
        true_residual = self._rc_residual(true_t_seq, out_driver, occ_driver, true_hvac_driver)
        pred_loss = self._huber_zero(pred_residual)
        true_loss = self._huber_zero(true_residual)
        k, k_out = self._stable_k(pred_residual.device, pred_residual.dtype)
        capacity = torch.nn.functional.softplus(self.raw_capacity).to(dtype=pred_loss.dtype) + self.eps
        hvac_gain = torch.nn.functional.softplus(self.raw_hvac).to(dtype=pred_loss.dtype)
        fan_coeffs = self._fan_mode_coefficients(pred_residual.device, pred_loss.dtype)
        occupancy_gain = torch.nn.functional.softplus(self.raw_occupancy).to(dtype=pred_loss.dtype)
        internal_g = torch.nn.functional.softplus(0.5 * (self.raw_internal + self.raw_internal.T)).to(dtype=pred_loss.dtype)
        internal_g = internal_g * (1.0 - torch.eye(self.num_rooms, device=internal_g.device, dtype=internal_g.dtype))
        bias_l2 = torch.linalg.vector_norm(self.bias.to(dtype=pred_loss.dtype), ord=2)
        bias_mse = torch.mean(self.bias.to(dtype=pred_loss.dtype) ** 2)
        raw_loss = pred_loss + true_loss
        pred_weighted = self.pred_weight * pred_loss
        true_weighted = self.true_weight * true_loss
        bias_penalty = self.bias_weight * bias_mse
        weighted_loss = pred_weighted + true_weighted + bias_penalty
        info = {
            "rc_loss": raw_loss.detach(),
            "rc_pred_loss": pred_loss.detach(),
            "rc_true_loss": true_loss.detach(),
            "rc_pred_weighted": pred_weighted.detach(),
            "rc_true_weighted": true_weighted.detach(),
            "rc_weighted": weighted_loss.detach(),
            "rc_bias_norm": bias_l2.detach(),
            "rc_bias_penalty": bias_penalty.detach(),
            "rc_capacity_mean": capacity.mean().detach(),
            "rc_capacity_min": capacity.min().detach(),
            "rc_k_out_mean": k_out.mean().detach(),
            "rc_internal_g_mean": internal_g.sum().detach() / (self.num_rooms * (self.num_rooms - 1)),
            "rc_hvac_gain_mean": hvac_gain.mean().detach(),
            "rc_fan_phi_1_mean": fan_coeffs[:, 1].mean().detach(),
            "rc_fan_phi_2_mean": fan_coeffs[:, 2].mean().detach(),
            "rc_fan_phi_3_mean": fan_coeffs[:, 3].mean().detach(),
            "rc_occ_gain_mean": occupancy_gain.mean().detach(),
            "rc_k_fro": torch.linalg.matrix_norm(k.detach(), ord="fro"),
        }
        return weighted_loss, info


def build_rc_loss_schema(input_cols, target_cols, context_steps, forecast_steps):
    rooms = sorted(
        {
            int(m.group(1))
            for col in target_cols
            for m in [re.match(r"y_room(\d+)_room_temp_t\d+$", str(col))]
            if m
        }
    )
    if not rooms:
        raise ValueError("RC loss requires target columns like y_room1_room_temp_t1.")

    target_lookup = {col: i for i, col in enumerate(target_cols)}
    input_lookup = {col: i for i, col in enumerate(input_cols)}
    occ_bases = ("occupant_num", "occupant_sitting", "occupant_walking", "occupant_standing")

    def require_target(name):
        if name not in target_lookup:
            raise ValueError(f"RC loss missing target column: {name}")
        return target_lookup[name]

    def require_input(name):
        if name not in input_lookup:
            raise ValueError(f"RC loss missing input column: {name}")
        return input_lookup[name]

    target_room_temp_idx = np.asarray(
        [
            [require_target(f"y_room{room}_room_temp_t{k}") for room in rooms]
            for k in range(1, forecast_steps + 1)
        ],
        dtype=np.int64,
    )
    target_outdoor_idx = np.asarray(
        [require_target(f"y_outdoor_temp_t{k}") for k in range(1, forecast_steps + 1)],
        dtype=np.int64,
    )
    target_supply_idx = np.asarray(
        [
            [require_target(f"y_room{room}_supply_temp_t{k}") for room in rooms]
            for k in range(1, forecast_steps + 1)
        ],
        dtype=np.int64,
    )
    current_room_temp_idx = np.asarray(
        [require_input(f"room{room}_room_temp_t{context_steps}") for room in rooms],
        dtype=np.int64,
    )
    current_outdoor_idx = int(require_input(f"outdoor_temp_t{context_steps}"))
    current_fan_idx = np.asarray(
        [require_input(f"room{room}_FCU_fan_feedback_t{context_steps}") for room in rooms],
        dtype=np.int64,
    )
    current_supply_idx = np.asarray(
        [require_input(f"room{room}_supply_temp_t{context_steps}") for room in rooms],
        dtype=np.int64,
    )

    target_occ_lists = []
    current_occ_lists = []
    max_occ = 1
    for k in range(1, forecast_steps + 1):
        step_lists = []
        for room in rooms:
            indices = [
                target_lookup[f"y_room{room}_{base}_t{k}"]
                for base in occ_bases
                if f"y_room{room}_{base}_t{k}" in target_lookup
            ]
            step_lists.append(indices)
            max_occ = max(max_occ, len(indices))
        target_occ_lists.append(step_lists)
    for room in rooms:
        indices = [
            input_lookup[f"room{room}_{base}_t{context_steps}"]
            for base in occ_bases
            if f"room{room}_{base}_t{context_steps}" in input_lookup
        ]
        current_occ_lists.append(indices)
        max_occ = max(max_occ, len(indices))

    target_occupancy_idx = np.full((forecast_steps, len(rooms), max_occ), -1, dtype=np.int64)
    target_occupancy_mask = np.zeros((forecast_steps, len(rooms), max_occ), dtype=np.int64)
    for k, step_lists in enumerate(target_occ_lists):
        for r, indices in enumerate(step_lists):
            target_occupancy_idx[k, r, : len(indices)] = indices
            target_occupancy_mask[k, r, : len(indices)] = 1

    current_occupancy_idx = np.full((len(rooms), max_occ), -1, dtype=np.int64)
    current_occupancy_mask = np.zeros((len(rooms), max_occ), dtype=np.int64)
    for r, indices in enumerate(current_occ_lists):
        current_occupancy_idx[r, : len(indices)] = indices
        current_occupancy_mask[r, : len(indices)] = 1

    return {
        "num_rooms": len(rooms),
        "rooms": rooms,
        "forecast_steps": forecast_steps,
        "condition_dim": len(input_cols),
        "target_room_temp_idx": target_room_temp_idx,
        "target_outdoor_idx": target_outdoor_idx,
        "target_supply_idx": target_supply_idx,
        "target_occupancy_idx": target_occupancy_idx,
        "target_occupancy_mask": target_occupancy_mask,
        "current_room_temp_idx": current_room_temp_idx,
        "current_outdoor_idx": current_outdoor_idx,
        "current_fan_idx": current_fan_idx,
        "current_supply_idx": current_supply_idx,
        "current_occupancy_idx": current_occupancy_idx,
        "current_occupancy_mask": current_occupancy_mask,
    }


def prepare_rc_loss_args(args, input_cols, target_cols, context_steps, forecast_steps):
    args.rc_loss_schema = build_rc_loss_schema(input_cols, target_cols, context_steps, forecast_steps)
    args.rc_target_mins, args.rc_target_maxs = _mins_maxs_for_columns(target_cols)
    args.rc_condition_mins, args.rc_condition_maxs = _mins_maxs_for_columns(input_cols)
    return args.rc_loss_schema


def attach_rc_loss(diffusion, args):
    if getattr(args, "rc_loss_weight", 0.0) <= 0 and getattr(args, "rc_true_weight", 0.0) <= 0:
        return diffusion
    if not hasattr(args, "rc_loss_schema"):
        raise ValueError("RC loss is enabled but the RC schema has not been prepared.")
    diffusion.physics_loss = MultiZoneRCLoss(
        args.rc_loss_schema,
        args.rc_target_mins,
        args.rc_target_maxs,
        args.rc_condition_mins,
        args.rc_condition_maxs,
        weight=args.rc_loss_weight,
        dt=args.rc_dt,
        huber_beta=args.rc_huber_beta,
        true_weight=args.rc_true_weight,
        bias_weight=args.rc_bias_weight,
        return_driver=args.rc_return_driver,
    )
    return diffusion


# -----------------------------------------------------------------------------
# Metrics and plotting
# -----------------------------------------------------------------------------

def extract_semantic_type_from_col(col_name):
    base = re.sub(r"_t\d+$", "", col_name.replace("y_", "", 1))
    return re.sub(r"^room\d+_", "", base)


def parse_target_loss_weights(spec):
    if not spec:
        return {}
    if isinstance(spec, dict):
        return {str(k).lower(): float(v) for k, v in spec.items()}

    weights = {}
    for item in re.split(r"[;,]", str(spec)):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        elif ":" in item:
            key, value = item.split(":", 1)
        else:
            raise ValueError(
                f"Invalid target loss weight '{item}'. Use semantic=weight, e.g. room_temp=2."
            )
        key = key.strip().lower()
        weight = float(value.strip())
        if not key or weight <= 0:
            raise ValueError(f"Invalid target loss weight '{item}'. Weights must be positive.")
        weights[key] = weight
    return weights


def build_action_loss_weights(target_cols, spec):
    semantic_weights = parse_target_loss_weights(spec)
    if not semantic_weights:
        return None, {}

    action_loss_weights = {}
    matched_counts = {key: 0 for key in semantic_weights}
    for i, col in enumerate(target_cols):
        semantic = extract_semantic_type_from_col(col).lower()
        if semantic in semantic_weights:
            weight = semantic_weights[semantic]
            action_loss_weights[i] = weight
            matched_counts[semantic] += 1

    unmatched = [key for key, count in matched_counts.items() if count == 0]
    if unmatched:
        available = sorted({extract_semantic_type_from_col(col).lower() for col in target_cols})
        raise ValueError(
            f"Unknown target loss weight semantic(s): {unmatched}. Available semantics: {available}"
        )

    return action_loss_weights, matched_counts


def extract_time_step_from_col(col_name):
    m = re.search(r"_t(\d+)$", str(col_name))
    return int(m.group(1)) if m else None


def is_discrete_integer_semantic(semantic: str) -> bool:
    return any(
        keyword in semantic.lower()
        for keyword in (
            "occupant_sitting",
            "occupant_walking",
            "occupant_standing",
            "occupant_num",
            "sitting",
            "walking",
            "standing",
        )
    )


def calculate_semantic_metrics(y_true, y_pred, target_cols, denormalize=True):
    if target_cols is None or len(target_cols) != y_true.shape[1]:
        return None
    if denormalize:
        y_true_eval = _denormalize_matrix(y_true, target_cols)
        y_pred_eval = _denormalize_matrix(y_pred, target_cols)
    else:
        y_true_eval, y_pred_eval = y_true, y_pred

    semantic_to_indices: Dict[str, List[int]] = {}
    for i, col in enumerate(target_cols):
        semantic_to_indices.setdefault(extract_semantic_type_from_col(col), []).append(i)

    metrics = {
        "mae": {},
        "mse": {},
        "cvrmse": {},
        "exact_match_rate": {},
        "within_1_rate": {},
        "within_2_rate": {},
        "classification_accuracy": {},
        "median_ae": {},
    }
    for semantic, indices in semantic_to_indices.items():
        yt = y_true_eval[:, indices].reshape(-1)
        yp = y_pred_eval[:, indices].reshape(-1)
        mae = mean_absolute_error(yt, yp)
        mse = mean_squared_error(yt, yp)
        rmse = math.sqrt(mse)
        mean_true = np.mean(np.abs(yt))
        metrics["mae"][semantic] = mae
        metrics["mse"][semantic] = mse
        metrics["cvrmse"][semantic] = (rmse / (mean_true + 1e-8)) * 100 if mean_true > 0 else 0.0
        metrics["median_ae"][semantic] = np.median(np.abs(yt - yp))

        if is_discrete_integer_semantic(semantic):
            yt_int = np.round(np.clip(yt, 0, None)).astype(int)
            yp_int = np.round(np.clip(yp, 0, None)).astype(int)
            abs_err = np.abs(yt_int - yp_int)
            metrics["exact_match_rate"][semantic] = float(np.mean(yt_int == yp_int) * 100)
            metrics["within_1_rate"][semantic] = float(np.mean(abs_err <= 1) * 100)
            metrics["within_2_rate"][semantic] = float(np.mean(abs_err <= 2) * 100)
            metrics["classification_accuracy"][semantic] = metrics["exact_match_rate"][semantic]
        else:
            metrics["exact_match_rate"][semantic] = None
            metrics["within_1_rate"][semantic] = None
            metrics["within_2_rate"][semantic] = None
            metrics["classification_accuracy"][semantic] = None
    return metrics


def calculate_per_timestep_metrics(y_true, y_pred, target_cols, denormalize=True):
    if target_cols is None or len(target_cols) != y_true.shape[1]:
        return None
    if denormalize:
        y_true_eval = _denormalize_matrix(y_true, target_cols)
        y_pred_eval = _denormalize_matrix(y_pred, target_cols)
    else:
        y_true_eval, y_pred_eval = y_true, y_pred

    timestep_to_indices: Dict[int, List[int]] = {}
    for i, col in enumerate(target_cols):
        k = extract_time_step_from_col(col)
        if k is not None:
            timestep_to_indices.setdefault(k, []).append(i)

    metrics = {"mae": {}, "mse": {}}
    for k, indices in sorted(timestep_to_indices.items()):
        yt = y_true_eval[:, indices].reshape(-1)
        yp = y_pred_eval[:, indices].reshape(-1)
        metrics["mae"][k] = mean_absolute_error(yt, yp)
        metrics["mse"][k] = mean_squared_error(yt, yp)
    return metrics


def evaluate_arrays(y_true, y_pred, target_cols):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred, multioutput="uniform_average")
    if CVRMSE_ON_ORIGINAL:
        yt = _denormalize_matrix(y_true, target_cols)
        yp = _denormalize_matrix(y_pred, target_cols)
        rmse_for_cvrmse = math.sqrt(mean_squared_error(yt, yp))
        mean_true = np.mean(np.abs(yt))
    else:
        rmse_for_cvrmse = rmse
        mean_true = np.mean(np.abs(y_true))
    cvrmse = (rmse_for_cvrmse / (mean_true + 1e-8)) * 100 if mean_true > 0 else 0.0

    semantic_metrics = calculate_semantic_metrics(y_true, y_pred, target_cols, denormalize=CVRMSE_ON_ORIGINAL)
    cvrmse_per_semantic = semantic_metrics["cvrmse"] if semantic_metrics else {}
    occupancy_emrs = [
        value
        for semantic, value in semantic_metrics.get("exact_match_rate", {}).items()
        if semantic_metrics and is_discrete_integer_semantic(semantic) and value is not None
    ]
    occupancy_emr = float(np.mean(occupancy_emrs)) if occupancy_emrs else None
    return mae, rmse, r2, cvrmse, cvrmse_per_semantic, occupancy_emr, semantic_metrics


def plot_training_curves(train_losses, val_rmse_history, semantic_metrics_history, timestep_metrics_history, plot_dir):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss - DiffusionWM")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "global_DiffusionWM_training_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_rmse_history, label="Validation RMSE", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Validation RMSE - DiffusionWM")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "global_DiffusionWM_val_rmse.png", dpi=300, bbox_inches="tight")
    plt.close()

    if semantic_metrics_history and semantic_metrics_history[0] is not None:
        semantics = sorted({s for m in semantic_metrics_history for s in m.get("mae", {}).keys()})
        for metric_name in ("mae", "mse", "cvrmse"):
            plt.figure(figsize=(12, 8))
            for semantic in semantics:
                values = [m.get(metric_name, {}).get(semantic, np.nan) for m in semantic_metrics_history]
                plt.plot(epochs, values, label=semantic.replace("_", " ").title(), linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel(metric_name.upper())
            plt.title(f"{metric_name.upper()} per Semantic Type - DiffusionWM")
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=9, ncol=2)
            plt.tight_layout()
            plt.savefig(plot_dir / f"global_DiffusionWM_{metric_name}_per_semantic.png", dpi=300, bbox_inches="tight")
            plt.close()

    if timestep_metrics_history and timestep_metrics_history[0] is not None:
        timesteps = sorted({k for m in timestep_metrics_history for k in m.get("mae", {}).keys()})
        for metric_name in ("mae", "mse"):
            plt.figure(figsize=(10, 6))
            for k in timesteps:
                values = [m.get(metric_name, {}).get(k, np.nan) for m in timestep_metrics_history]
                plt.plot(epochs, values, label=f"t+{k}", linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel(metric_name.upper())
            plt.title(f"{metric_name.upper()} per Time Step - DiffusionWM")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / f"global_DiffusionWM_{metric_name}_per_timestep.png", dpi=300, bbox_inches="tight")
            plt.close()


def save_preds(pred_dir, y_true, y_pred, target_cols, suffix):
    data = {}
    cols = []
    for i, col in enumerate(target_cols):
        for prefix, values in (
            ("y_true", y_true[:, i]),
            ("y_pred", y_pred[:, i]),
            ("abs_error", np.abs(y_true[:, i] - y_pred[:, i])),
        ):
            name = f"{prefix}_{col}"
            data[name] = values
            cols.append(name)
    out = pred_dir / f"global_DiffusionWM_preds_{suffix}.csv"
    pd.DataFrame(data, columns=cols).to_csv(out, index=False)
    return out


def semantic_metrics_to_rows(metrics, model_name, suffix, epoch=None):
    if not metrics:
        return []
    semantics = sorted(metrics.get("mae", {}).keys())
    rows = []
    for semantic in semantics:
        rows.append(
            {
                "Room": "global",
                "Model": model_name,
                "Split": suffix,
                "Epoch": epoch,
                "Semantic": semantic,
                "MAE": metrics.get("mae", {}).get(semantic),
                "MSE": metrics.get("mse", {}).get(semantic),
                "CVRMSE": metrics.get("cvrmse", {}).get(semantic),
                "MedianAE": metrics.get("median_ae", {}).get(semantic),
                "ExactMatchRate": metrics.get("exact_match_rate", {}).get(semantic),
                "Within1Rate": metrics.get("within_1_rate", {}).get(semantic),
                "Within2Rate": metrics.get("within_2_rate", {}).get(semantic),
                "ClassificationAccuracy": metrics.get("classification_accuracy", {}).get(semantic),
            }
        )
    return rows


def save_semantic_metrics(out_dir, metrics, model_name, suffix):
    rows = semantic_metrics_to_rows(metrics, model_name, suffix)
    out = out_dir / f"global_{model_name}_semantic_metrics_{suffix}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _fmt_metric(value, suffix=""):
    if value is None:
        return "N/A"
    return f"{value:.4f}{suffix}"


def log_semantic_metrics(metrics, model_name, suffix):
    if not metrics:
        return
    logger.info("%s (%s) test semantic metrics:", model_name, suffix)
    semantics = sorted(metrics.get("mae", {}).keys())
    for semantic in semantics:
        logger.info(
            "  %s | MAE=%s | MSE=%s | CVRMSE=%s | MedianAE=%s | EMR=%s | Within1=%s | Within2=%s | Acc=%s",
            semantic,
            _fmt_metric(metrics.get("mae", {}).get(semantic)),
            _fmt_metric(metrics.get("mse", {}).get(semantic)),
            _fmt_metric(metrics.get("cvrmse", {}).get(semantic), "%"),
            _fmt_metric(metrics.get("median_ae", {}).get(semantic)),
            _fmt_metric(metrics.get("exact_match_rate", {}).get(semantic), "%"),
            _fmt_metric(metrics.get("within_1_rate", {}).get(semantic), "%"),
            _fmt_metric(metrics.get("within_2_rate", {}).get(semantic), "%"),
            _fmt_metric(metrics.get("classification_accuracy", {}).get(semantic), "%"),
        )


def save_semantic_history(out_dir, semantic_metrics_history, model_name):
    rows = []
    for epoch, metrics in enumerate(semantic_metrics_history, start=1):
        rows.extend(semantic_metrics_to_rows(metrics, model_name, "val", epoch=epoch))
    out = out_dir / f"global_{model_name}_semantic_metrics_history.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def timestep_metrics_to_rows(metrics, model_name, suffix, epoch=None):
    if not metrics:
        return []
    timesteps = sorted(metrics.get("mae", {}).keys())
    rows = []
    for timestep in timesteps:
        rows.append(
            {
                "Room": "global",
                "Model": model_name,
                "Split": suffix,
                "Epoch": epoch,
                "Timestep": timestep,
                "MAE": metrics.get("mae", {}).get(timestep),
                "MSE": metrics.get("mse", {}).get(timestep),
            }
        )
    return rows


def save_timestep_metrics(out_dir, metrics, model_name, suffix):
    rows = timestep_metrics_to_rows(metrics, model_name, suffix)
    out = out_dir / f"global_{model_name}_timestep_metrics_{suffix}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


# -----------------------------------------------------------------------------
# Original diffuser construction and sampling
# -----------------------------------------------------------------------------

def build_original_diffusion(action_dim: int, observation_dim: int, args, save_configs: bool = False):
    model_savepath = (str(args.savepath), "model_config.pkl") if save_configs else None
    diffusion_savepath = (str(args.savepath), "diffusion_config.pkl") if save_configs else None
    z_dim = args.z_dim
    model_config = utils.Config(
        models.TemporalUnet,
        savepath=model_savepath,
        horizon=args.horizon,
        transition_dim=action_dim + observation_dim + z_dim,
        cond_dim=observation_dim + z_dim,
        dim_mults=args.dim_mults,
        attention=args.attention,
        device=args.device,
    )
    diffusion_config = utils.Config(
        models.GaussianDiffusion,
        savepath=diffusion_savepath,
        horizon=args.horizon,
        observation_dim=observation_dim + z_dim,
        action_dim=action_dim,
        n_timesteps=args.n_diffusion_steps,
        loss_type=args.loss_type,
        clip_denoised=args.clip_denoised,
        predict_epsilon=args.predict_epsilon,
        action_weight=args.action_weight,
        loss_discount=args.loss_discount,
        loss_weights=None,
        action_loss_weights=args.action_loss_weights,
        smoothl1_beta=args.smoothl1_beta,
        device=args.device,
    )
    model = model_config()
    diffusion = diffusion_config(model)
    diffusion = attach_rc_loss(diffusion, args)
    return diffusion


def build_original_trainer(diffusion, dataset, args, ckpt_dir):
    trainer_config = utils.Config(
        utils.Trainer,
        savepath=(str(args.savepath), "trainer_config.pkl"),
        ema_decay=args.ema_decay,
        train_batch_size=args.batch_size,
        train_lr=args.learning_rate,
        gradient_accumulate_every=args.gradient_accumulate_every,
        sample_freq=args.sample_freq,
        save_freq=args.save_freq,
        label_freq=int(args.n_train_steps // args.n_saves),
        save_parallel=args.save_parallel,
        results_folder=str(ckpt_dir),
        bucket=args.bucket,
        n_reference=args.n_reference,
        use_latent_encoder=args.use_latent_encoder,
        z_dim=args.z_dim,
        constant_z_value=args.constant_z_value,
    )
    return trainer_config(diffusion, dataset, NullRenderer())


@torch.no_grad()
def sample_with_original_models(
    diffusion,
    model_p,
    cond_tensor,
    sample_batch_size=256,
    deterministic_z=True,
    use_latent_encoder=True,
    z_dim=16,
    constant_z_value=0.0,
):
    diffusion.eval()
    diffusion.clip_denoised = True
    if use_latent_encoder:
        model_p.eval()
    outs = []
    for i in range(0, cond_tensor.shape[0], sample_batch_size):
        cond = cond_tensor[i : i + sample_batch_size].to(DEVICE)
        if use_latent_encoder:
            mu_p, logvar_p = model_p(cond)
            z = mu_p if deterministic_z else reparameterize(mu_p, logvar_p)
        else:
            z = torch.full(
                (cond.shape[0], z_dim),
                constant_z_value,
                dtype=cond.dtype,
                device=cond.device,
            )
        cond_cat = torch.cat([cond, z], dim=-1)
        cond_dict = {t: cond_cat for t in range(diffusion.horizon)}
        sample = diffusion.conditional_sample(cond_dict, horizon=diffusion.horizon, verbose=False)
        preds = sample.trajectories[:, :, : diffusion.action_dim].reshape(cond.shape[0], -1)
        outs.append(preds.detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def clone_eval_models(trainer, action_dim, observation_dim, args, use_ema=True):
    diffusion = build_original_diffusion(action_dim, observation_dim, args).to(DEVICE)
    model_p = None
    if args.use_latent_encoder:
        model_p = trainer.ema_model_p if use_ema else trainer.model_p
    if use_ema:
        diffusion.load_state_dict(trainer.ema_model.state_dict())
    else:
        diffusion.load_state_dict(trainer.model.state_dict())
    return diffusion, model_p


def save_checkpoint(path, trainer, rmse=None, epoch=None):
    payload = {
        "epoch": epoch,
        "step": trainer.step,
        "rmse": rmse,
        "model": trainer.model.state_dict(),
        "ema": trainer.ema_model.state_dict(),
        "use_latent_encoder": trainer.use_latent_encoder,
        "z_dim": trainer.z_dim,
        "constant_z_value": trainer.constant_z_value,
    }
    if trainer.use_latent_encoder:
        payload.update(
            {
                "model_p": trainer.model_p.state_dict(),
                "model_p_ema": trainer.ema_model_p.state_dict(),
                "model_q": trainer.model_q.state_dict(),
                "model_q_ema": trainer.ema_model_q.state_dict(),
            }
        )
    torch.save(payload, path)


def _to_serializable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_to_serializable(v) for v in value]
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    return value


def _format_yaml_scalar(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def _write_yaml_value(f, key, value, indent=0):
    prefix = " " * indent
    if isinstance(value, dict):
        f.write(f"{prefix}{key}:\n")
        for child_key, child_value in value.items():
            _write_yaml_value(f, child_key, child_value, indent + 2)
    elif isinstance(value, list):
        f.write(f"{prefix}{key}:\n")
        for item in value:
            item_prefix = " " * (indent + 2)
            if isinstance(item, dict):
                f.write(f"{item_prefix}-\n")
                for child_key, child_value in item.items():
                    _write_yaml_value(f, child_key, child_value, indent + 4)
            elif isinstance(item, list):
                f.write(f"{item_prefix}- {json.dumps(item, ensure_ascii=False)}\n")
            else:
                f.write(f"{item_prefix}- {_format_yaml_scalar(item)}\n")
    else:
        f.write(f"{prefix}{key}: {_format_yaml_scalar(value)}\n")


def save_train_config(out_dir, args, T, N, F_in, input_cols, target_cols):
    all_args = {key: _to_serializable(value) for key, value in sorted(vars(args).items())}
    config = {
        "train_config": {
            "SEED": args.seed,
            "TEST_SIZE": args.test_size,
            "BATCH_SIZE": args.batch_size,
            "EPOCHS": args.epochs,
            "N_TRAIN_STEPS": args.n_train_steps,
            "N_STEPS_PER_EPOCH": args.n_steps_per_epoch,
            "LR": args.learning_rate,
            "WEIGHT_DECAY": args.weight_decay,
            "EARLY_PATIENCE": args.early_patience,
            "MIN_DELTA_RMSE": args.min_delta_rmse,
            "SAVE_INTERVAL": args.save_interval,
            "LOG_INTERVAL": args.log_interval,
            "SAVE_FREQ": args.save_freq,
            "SAMPLE_FREQ": args.sample_freq,
            "N_SAVES": args.n_saves,
            "SAVE_PARALLEL": args.save_parallel,
            "N_REFERENCE": args.n_reference,
            "SAMPLE_BATCH_SIZE": args.sample_batch_size,
            "DETERMINISTIC_Z": args.deterministic_z,
            "BUCKET": args.bucket,
            "DEVICE": args.device,
        },
        "diffusion_config": {
            "USES_ORIGINAL_TRAINER": True,
            "TRAINER": "diffuser.utils.Trainer",
            "MODEL": "diffuser.models.TemporalUnet",
            "DIFFUSION": "diffuser.models.GaussianDiffusion",
            "WM_HORIZON_MODE": args.wm_horizon_mode,
            "HORIZON": args.horizon,
            "N_DIFFUSION_STEPS": args.n_diffusion_steps,
            "CLIP_DENOISED": args.clip_denoised,
            "PREDICT_EPSILON": args.predict_epsilon,
            "LOSS_TYPE": args.loss_type,
            "SMOOTHL1_BETA": args.smoothl1_beta,
            "ACTION_WEIGHT": args.action_weight,
            "LOSS_DISCOUNT": args.loss_discount,
            "TARGET_LOSS_WEIGHTS": args.target_loss_weights,
            "TARGET_LOSS_WEIGHT_MATCHES": args.target_loss_weight_matches,
            "RC_LOSS_WEIGHT": args.rc_loss_weight,
            "RC_DT": args.rc_dt,
            "RC_HUBER_BETA": args.rc_huber_beta,
            "RC_TRUE_WEIGHT": args.rc_true_weight,
            "RC_BIAS_WEIGHT": args.rc_bias_weight,
            "RC_RETURN_DRIVER": args.rc_return_driver,
            "DIM_MULTS": list(args.dim_mults),
            "ATTENTION": args.attention,
            "EMA_DECAY": args.ema_decay,
            "GRADIENT_ACCUMULATE_EVERY": args.gradient_accumulate_every,
            "USE_LATENT_ENCODER": args.use_latent_encoder,
            "Z_DIM": args.z_dim,
            "CONSTANT_Z_VALUE": args.constant_z_value,
        },
        "data_config": {
            "dataset": str(args.dataset),
            "T": T,
            "N": N,
            "F_in": F_in,
            "input_dim_flat": len(input_cols),
            "target_dim_flat": len(target_cols),
            "input_cols": input_cols,
            "target_cols": target_cols,
        },
        "all_args": all_args,
    }
    json_path = out_dir / "train_config.json"
    txt_path = out_dir / "train_config.txt"
    yaml_path = out_dir / "train_config.yaml"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        for section, values in config.items():
            f.write(f"{section}\n")
            f.write("-" * 80 + "\n")
            for key, value in values.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
    with open(yaml_path, "w", encoding="utf-8") as f:
        for section, values in config.items():
            _write_yaml_value(f, section, values)
    return json_path, txt_path, yaml_path


# -----------------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------------

def parse_dim_mults(value: str):
    if isinstance(value, tuple):
        return value
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return tuple(int(p) for p in parts) if parts else DIM_MULTS


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def add_bool_arg(parser, name, default):
    parser.add_argument(name, type=str2bool, nargs="?", const=True, default=default)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a diffusion world model using the repository's original Trainer/GaussianDiffusion."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--test_size", type=float, default=TEST_SIZE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--n_steps_per_epoch", "--steps_per_epoch", dest="n_steps_per_epoch", type=int, default=N_STEPS_PER_EPOCH)
    parser.add_argument("--learning_rate", "--lr", dest="learning_rate", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--early_patience", type=int, default=EARLY_PATIENCE)
    parser.add_argument("--min_delta_rmse", type=float, default=MIN_DELTA_RMSE)
    parser.add_argument("--save_interval", type=int, default=SAVE_INTERVAL)
    parser.add_argument("--log_interval", type=int, default=LOG_INTERVAL)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument(
        "--wm_horizon_mode",
        choices=["temporal", "flat"],
        default=WM_HORIZON_MODE,
        help="temporal uses diffusion horizon=N and per-step targets; flat keeps the old horizon=1 flattened target layout.",
    )
    parser.add_argument("--n_diffusion_steps", type=int, default=N_DIFFUSION_STEPS)
    add_bool_arg(parser, "--clip_denoised", False)
    add_bool_arg(parser, "--predict_epsilon", PREDICT_EPSILON)
    parser.add_argument("--loss_type", choices=["l1", "l2", "smoothl1"], default=LOSS_TYPE)
    parser.add_argument("--smoothl1_beta", type=float, default=SMOOTHL1_BETA)
    parser.add_argument("--action_weight", type=float, default=ACTION_WEIGHT)
    parser.add_argument("--loss_discount", type=float, default=LOSS_DISCOUNT)
    parser.add_argument(
        "--target_loss_weights",
        default=TARGET_LOSS_WEIGHTS,
        help="Comma-separated semantic weights for predicted targets, e.g. room_temp=2,return_temp=2.",
    )
    parser.add_argument(
        "--rc_loss_weight",
        type=float,
        default=RC_LOSS_WEIGHT,
        help="Absolute weight for the predicted-trajectory RC residual.",
    )
    parser.add_argument("--rc_dt", type=float, default=RC_DT)
    parser.add_argument("--rc_huber_beta", type=float, default=RC_HUBER_BETA)
    parser.add_argument(
        "--rc_true_weight",
        type=float,
        default=RC_TRUE_WEIGHT,
        help="Absolute weight for the true-trajectory RC residual used to calibrate RC parameters.",
    )
    parser.add_argument("--rc_bias_weight", type=float, default=RC_BIAS_WEIGHT)
    parser.add_argument(
        "--rc_return_driver",
        choices=["pred", "true", "pred_detached"],
        default=RC_RETURN_DRIVER,
        help="Return-temperature driver used in predicted RC residual. 'pred' lets RC loss backprop to return_temp.",
    )
    parser.add_argument("--dim_mults", type=parse_dim_mults, default=DIM_MULTS)
    add_bool_arg(parser, "--attention", ATTENTION)
    parser.add_argument("--ema_decay", type=float, default=EMA_DECAY)
    parser.add_argument("--gradient_accumulate_every", type=int, default=GRADIENT_ACCUMULATE_EVERY)
    add_bool_arg(parser, "--use_latent_encoder", USE_LATENT_ENCODER)
    parser.add_argument("--z_dim", type=int, default=None)
    parser.add_argument("--constant_z_value", type=float, default=CONSTANT_Z_VALUE)
    parser.add_argument("--n_train_steps", type=float, default=None)
    parser.add_argument("--n_saves", type=int, default=5)
    parser.add_argument("--sample_freq", type=int, default=0)
    parser.add_argument("--save_freq", type=int, default=10000)
    add_bool_arg(parser, "--save_parallel", False)
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--n_reference", type=int, default=8)
    parser.add_argument("--sample_batch_size", type=int, default=256)
    add_bool_arg(parser, "--deterministic_z", True)
    parser.add_argument("--device", default=str(DEVICE))
    args = parser.parse_args()
    if args.z_dim is None:
        args.z_dim = 16 if args.use_latent_encoder else Z_DIM
    if args.use_latent_encoder and args.z_dim != 16:
        raise ValueError("When --use_latent_encoder true, use --z_dim 16 to match the original encoder setup.")
    if args.smoothl1_beta <= 0:
        raise ValueError("--smoothl1_beta must be positive.")
    if args.rc_loss_weight < 0:
        raise ValueError("--rc_loss_weight must be non-negative.")
    if args.rc_dt <= 0:
        raise ValueError("--rc_dt must be positive.")
    if args.rc_huber_beta <= 0:
        raise ValueError("--rc_huber_beta must be positive.")
    if args.rc_true_weight < 0:
        raise ValueError("--rc_true_weight must be non-negative.")
    if args.rc_bias_weight < 0:
        raise ValueError("--rc_bias_weight must be non-negative.")
    if args.n_train_steps is None:
        args.n_train_steps = args.epochs * args.n_steps_per_epoch
    args.action_loss_weights = None
    args.target_loss_weight_matches = {}
    return args


def setup_logging(log_dir):
    log_file = log_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_file


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    global DEVICE
    DEVICE = torch.device(args.device)
    utils.arrays.DEVICE = args.device
    set_seed(args.seed)

    out_dir, pred_dir, ckpt_dir, plot_dir, log_dir = make_out_dirs("DiffusionWM")
    args.savepath = str(ckpt_dir)
    log_file = setup_logging(log_dir)

    files = sorted(glob.glob(str(args.dataset)))
    if not files:
        raise SystemExit(f"No dataset found: {args.dataset}")

    results = []
    test_semantic_rows = []
    test_timestep_rows = []
    for fpath in files:
        csv_path = Path(fpath)
        logger.info("Dataset: %s", csv_path)
        df = pd.read_csv(csv_path)
        T, N, input_bases, target_bases, input_cols, target_cols, manifest_path = build_columns(df, csv_path)
        F_in = len(input_cols) // T
        logger.info(
            "Using T=%d, N=%d, F_in=%d, input_dim=%d, target_dim=%d, manifest=%s",
            T,
            N,
            F_in,
            len(input_cols),
            len(target_cols),
            manifest_path,
        )
        model_horizon = N if args.wm_horizon_mode == "temporal" else 1
        args.horizon = model_horizon
        if args.use_latent_encoder and args.horizon != 1:
            raise ValueError("--use_latent_encoder is only supported with --wm_horizon_mode flat.")
        if len(target_cols) % args.horizon != 0:
            raise ValueError(f"Target dim {len(target_cols)} is not divisible by horizon {args.horizon}.")
        target_step_dim = len(target_cols) // args.horizon
        target_cols_for_loss_weights = target_cols[:target_step_dim] if args.wm_horizon_mode == "temporal" else target_cols
        args.action_loss_weights, args.target_loss_weight_matches = build_action_loss_weights(
            target_cols_for_loss_weights, args.target_loss_weights
        )
        if args.action_loss_weights:
            logger.info(
                "Target loss weights enabled: %s (matched target columns: %s)",
                args.target_loss_weights,
                args.target_loss_weight_matches,
            )
        logger.info(
            "World-model diffusion layout: mode=%s, horizon=%d, action_dim_per_step=%d",
            args.wm_horizon_mode,
            args.horizon,
            target_step_dim,
        )
        if args.rc_loss_weight > 0 or args.rc_true_weight > 0:
            rc_schema = prepare_rc_loss_args(args, input_cols, target_cols, T, N)
            logger.info(
                "Multi-zone RC loss enabled: rooms=%s, pred_weight=%.6g, true_weight=%.6g, dt=%.6g, huber_beta=%.6g",
                rc_schema["rooms"],
                args.rc_loss_weight,
                args.rc_true_weight,
                args.rc_dt,
                args.rc_huber_beta,
            )

        X_flat = _normalize_matrix(df[input_cols].astype(np.float32).values, input_cols).astype(np.float32)
        y_flat = _normalize_matrix(df[target_cols].astype(np.float32).values, target_cols).astype(np.float32)

        Xtr, Xte, ytr, yte = train_test_split(X_flat, y_flat, test_size=args.test_size, random_state=args.seed)
        train_dataset = WindowedWorldModelDataset(Xtr, ytr, horizon=args.horizon)
        val_cond = torch.tensor(Xte, dtype=torch.float32)

        diffusion = build_original_diffusion(
            train_dataset.action_dim,
            train_dataset.observation_dim,
            args,
            save_configs=True,
        ).to(DEVICE)
        trainer = build_original_trainer(diffusion, train_dataset, args, ckpt_dir)
        utils.training.Dim = str(args.dim_mults)
        utils.training.Horizon = str(args.horizon)
        utils.training.n_diffusion_steps = str(args.n_diffusion_steps)

        total_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        logger.info("Original diffusion model parameters: %d (%.2fM)", total_params, total_params / 1e6)

        steps_per_epoch = args.n_steps_per_epoch
        best_rmse = float("inf")
        best_snapshot = None
        patience = 0
        train_losses = []
        val_rmse_history = []
        semantic_metrics_history = []
        timestep_metrics_history = []

        for epoch in range(args.epochs):
            train_log = trainer.train(n_train_steps=steps_per_epoch)
            train_loss = float(train_log.get("diffusion_loss_mean", np.nan))
            train_losses.append(train_loss)

            eval_diffusion, eval_model_p = clone_eval_models(
                trainer, train_dataset.action_dim, train_dataset.observation_dim, args, use_ema=True
            )
            y_pred = sample_with_original_models(
                eval_diffusion,
                eval_model_p,
                val_cond,
                sample_batch_size=args.sample_batch_size,
                deterministic_z=args.deterministic_z,
                use_latent_encoder=args.use_latent_encoder,
                z_dim=args.z_dim,
                constant_z_value=args.constant_z_value,
            )
            mae, rmse, r2, cvrmse, cvrmse_per_semantic, occupancy_emr, semantic_metrics = evaluate_arrays(
                yte, y_pred, target_cols
            )
            timestep_metrics = calculate_per_timestep_metrics(
                yte, y_pred, target_cols, denormalize=CVRMSE_ON_ORIGINAL
            )
            val_rmse_history.append(rmse)
            semantic_metrics_history.append(semantic_metrics)
            timestep_metrics_history.append(timestep_metrics)

            if epoch == 0 or (epoch + 1) % args.log_interval == 0:
                occupancy_str = f"{occupancy_emr:.4f}%" if occupancy_emr is not None else "N/A"
                rc_loss = train_log.get("rc_loss_mean")
                rc_weighted = train_log.get("rc_weighted_mean")
                rc_str = ""
                if rc_loss is not None:
                    rc_weighted = 0.0 if rc_weighted is None else rc_weighted
                    rc_pred = train_log.get("rc_pred_loss_mean", 0.0)
                    rc_true = train_log.get("rc_true_loss_mean", 0.0)
                    rc_pred_weighted = train_log.get("rc_pred_weighted_mean", 0.0)
                    rc_true_weighted = train_log.get("rc_true_weighted_mean", 0.0)
                    rc_bias = train_log.get("rc_bias_norm_mean", 0.0)
                    rc_bias_penalty = train_log.get("rc_bias_penalty_mean", 0.0)
                    rc_cap = train_log.get("rc_capacity_mean_mean", 0.0)
                    rc_k_out = train_log.get("rc_k_out_mean_mean", 0.0)
                    rc_g = train_log.get("rc_internal_g_mean_mean", 0.0)
                    rc_hvac = train_log.get("rc_hvac_gain_mean_mean", 0.0)
                    rc_phi1 = train_log.get("rc_fan_phi_1_mean_mean", 0.0)
                    rc_phi2 = train_log.get("rc_fan_phi_2_mean_mean", 0.0)
                    rc_phi3 = train_log.get("rc_fan_phi_3_mean_mean", 0.0)
                    rc_occ = train_log.get("rc_occ_gain_mean_mean", 0.0)
                    rc_str = (
                        f" | rc_loss={float(rc_loss):.6f}"
                        f" | rc_pred={float(rc_pred):.6f}"
                        f" | rc_true={float(rc_true):.6f}"
                        f" | rc_pred_w={float(rc_pred_weighted):.6f}"
                        f" | rc_true_w={float(rc_true_weighted):.6f}"
                        f" | rc_d_norm={float(rc_bias):.6f}"
                        f" | rc_d_pen={float(rc_bias_penalty):.3e}"
                        f" | rc_C={float(rc_cap):.4f}"
                        f" | rc_kout={float(rc_k_out):.4f}"
                        f" | rc_g={float(rc_g):.4f}"
                        f" | rc_hvac={float(rc_hvac):.4f}"
                        f" | rc_phi=[{float(rc_phi1):.3f},{float(rc_phi2):.3f},{float(rc_phi3):.3f}]"
                        f" | rc_occ={float(rc_occ):.4f}"
                        f" | rc_weighted={float(rc_weighted):.6f}"
                    )
                logger.info(
                    "Epoch %03d/%03d | train_loss=%.6f%s | val_mae=%.6f | val_rmse=%.6f | r2=%.6f | cvrmse=%.4f%% | occupancy_emr=%s",
                    epoch + 1,
                    args.epochs,
                    train_loss,
                    rc_str,
                    mae,
                    rmse,
                    r2,
                    cvrmse,
                    occupancy_str,
                )

            if (epoch + 1) % args.save_interval == 0:
                ckpt_path = ckpt_dir / f"global_DiffusionWM_epoch_{epoch + 1}.pth"
                save_checkpoint(ckpt_path, trainer, rmse=rmse, epoch=epoch + 1)
                logger.info("Checkpoint saved: %s", ckpt_path)

            if best_rmse - rmse > args.min_delta_rmse:
                best_rmse = rmse
                best_snapshot = {
                    "model": {k: v.detach().cpu() for k, v in trainer.model.state_dict().items()},
                    "ema": {k: v.detach().cpu() for k, v in trainer.ema_model.state_dict().items()},
                    "use_latent_encoder": trainer.use_latent_encoder,
                    "z_dim": trainer.z_dim,
                    "constant_z_value": trainer.constant_z_value,
                    "epoch": epoch + 1,
                    "rmse": rmse,
                }
                if trainer.use_latent_encoder:
                    best_snapshot.update(
                        {
                            "model_p": {k: v.detach().cpu() for k, v in trainer.model_p.state_dict().items()},
                            "model_p_ema": {k: v.detach().cpu() for k, v in trainer.ema_model_p.state_dict().items()},
                            "model_q": {k: v.detach().cpu() for k, v in trainer.model_q.state_dict().items()},
                            "model_q_ema": {k: v.detach().cpu() for k, v in trainer.ema_model_q.state_dict().items()},
                        }
                    )
                patience = 0
            else:
                patience += 1
                if patience >= args.early_patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        final_path = ckpt_dir / "global_DiffusionWM_final.pth"
        save_checkpoint(final_path, trainer, rmse=val_rmse_history[-1], epoch=len(train_losses))
        logger.info("Final model saved: %s", final_path)

        best_path = None
        if best_snapshot is not None:
            best_path = ckpt_dir / "global_DiffusionWM_best.pth"
            torch.save(best_snapshot, best_path)
            logger.info("Best model saved: %s (RMSE=%.6f)", best_path, best_snapshot["rmse"])

        plot_training_curves(train_losses, val_rmse_history, semantic_metrics_history, timestep_metrics_history, plot_dir)
        semantic_history_path = save_semantic_history(out_dir, semantic_metrics_history, "DiffusionWM")
        logger.info("Semantic metrics history saved: %s", semantic_history_path)

        def evaluate_checkpoint(path, suffix):
            payload = torch.load(path, map_location=DEVICE)
            eval_diffusion = build_original_diffusion(train_dataset.action_dim, train_dataset.observation_dim, args).to(DEVICE)
            eval_diffusion.load_state_dict(payload["ema"] if "ema" in payload else payload["model"])
            eval_model_p = None
            if args.use_latent_encoder:
                eval_model_p = copy.deepcopy(trainer.ema_model_p).to(DEVICE)
            if args.use_latent_encoder and "model_p_ema" in payload:
                eval_model_p.load_state_dict(payload["model_p_ema"])
            y_pred_eval = sample_with_original_models(
                eval_diffusion,
                eval_model_p,
                val_cond,
                sample_batch_size=args.sample_batch_size,
                deterministic_z=args.deterministic_z,
                use_latent_encoder=args.use_latent_encoder,
                z_dim=args.z_dim,
                constant_z_value=args.constant_z_value,
            )
            mae, rmse, r2, cvrmse, cvrmse_per_semantic, occupancy_emr, semantic_metrics_eval = evaluate_arrays(
                yte, y_pred_eval, target_cols
            )
            timestep_metrics_eval = calculate_per_timestep_metrics(
                yte, y_pred_eval, target_cols, denormalize=CVRMSE_ON_ORIGINAL
            )
            pred_path = save_preds(pred_dir, yte, y_pred_eval, target_cols, suffix=suffix)
            semantic_path = save_semantic_metrics(out_dir, semantic_metrics_eval, "DiffusionWM", suffix)
            timestep_path = save_timestep_metrics(out_dir, timestep_metrics_eval, "DiffusionWM", suffix)
            test_semantic_rows.extend(
                semantic_metrics_to_rows(semantic_metrics_eval, f"DiffusionWM-{suffix.title()}", "test")
            )
            test_timestep_rows.extend(
                timestep_metrics_to_rows(timestep_metrics_eval, f"DiffusionWM-{suffix.title()}", "test")
            )
            cvrmse_str = ";".join(f"{k}:{v:.4f}" for k, v in sorted(cvrmse_per_semantic.items()))
            occupancy_str = f"{occupancy_emr:.4f}" if occupancy_emr is not None else "N/A"
            results.append(
                [
                    "global",
                    f"DiffusionWM-{suffix.title()}",
                    T,
                    F_in,
                    mae,
                    rmse,
                    r2,
                    cvrmse,
                    cvrmse_str,
                    occupancy_str,
                    str(pred_path),
                ]
            )
            logger.info(
                "DiffusionWM (%s) | MAE=%.6f RMSE=%.6f R2=%.6f CVRMSE=%.4f%% | Occupancy EMR=%s%% | preds=%s | semantic_metrics=%s | timestep_metrics=%s",
                suffix,
                mae,
                rmse,
                r2,
                cvrmse,
                occupancy_str,
                pred_path,
                semantic_path,
                timestep_path,
            )
            log_semantic_metrics(semantic_metrics_eval, "DiffusionWM", suffix)
            return eval_diffusion, eval_model_p

        eval_models = []
        if best_path is not None:
            eval_models.append(("best", *evaluate_checkpoint(best_path, "best")))
        eval_models.append(("final", *evaluate_checkpoint(final_path, "final")))

        X_full = torch.tensor(X_flat, dtype=torch.float32)
        for suffix, eval_diffusion, eval_model_p in eval_models:
            y_pred_full = sample_with_original_models(
                eval_diffusion,
                eval_model_p,
                X_full,
                sample_batch_size=args.sample_batch_size,
                deterministic_z=args.deterministic_z,
                use_latent_encoder=args.use_latent_encoder,
                z_dim=args.z_dim,
                constant_z_value=args.constant_z_value,
            )
            local_full = pred_dir / f"global_DiffusionWM_preds_full_{suffix}.csv"
            pd.DataFrame(y_pred_full, columns=[f"y_pred_{c}" for c in target_cols]).to_csv(local_full, index=False)

            shared_full = FULL_RESULTS_DIR / f"global__DiffusionWM__{suffix}__full.csv"
            shared_df = pd.DataFrame(
                {
                    "room": ["global"] * len(y_pred_full),
                    "model": [f"DiffusionWM-{suffix.title()}"] * len(y_pred_full),
                    "T": [T] * len(y_pred_full),
                    "F": [F_in] * len(y_pred_full),
                }
            )
            for i, col in enumerate(target_cols):
                shared_df[f"y_pred_{col}"] = y_pred_full[:, i]
            shared_df.to_csv(shared_full, index=False)
            logger.info("Saved full predictions (%s): %s and %s", suffix, local_full, shared_full)

        config_json, config_txt, config_yaml = save_train_config(out_dir, args, T, N, F_in, input_cols, target_cols)
        logger.info("Training config saved: %s, %s and %s", config_json, config_txt, config_yaml)

    if results:
        out_csv = out_dir / "diffusion_wm_results.csv"
        pd.DataFrame(
            results,
            columns=[
                "Room",
                "Model",
                "T",
                "F",
                "MAE",
                "RMSE",
                "R2",
                "CVRMSE_Overall",
                "CVRMSE_PerDim",
                "Occupancy_EMR",
                "preds_path",
            ],
        ).to_csv(out_csv, index=False)
        logger.info("Saved results to %s", out_csv)
        if test_semantic_rows:
            semantic_out_csv = out_dir / "diffusion_wm_test_semantic_results.csv"
            pd.DataFrame(test_semantic_rows).to_csv(semantic_out_csv, index=False)
            logger.info("Saved test semantic results to %s", semantic_out_csv)
        if test_timestep_rows:
            timestep_out_csv = out_dir / "diffusion_wm_test_timestep_results.csv"
            pd.DataFrame(test_timestep_rows).to_csv(timestep_out_csv, index=False)
            logger.info("Saved test timestep results to %s", timestep_out_csv)
        logger.info("Prediction CSVs in: %s", pred_dir)
        logger.info("Full-period predictions gathered in: %s", FULL_RESULTS_DIR)
        logger.info("Training log saved to: %s", log_file)
    else:
        logger.info("No models were trained.")


if __name__ == "__main__":
    main()
