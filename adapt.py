import os
import random
import sys
import time
import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence, Tuple
from collections.abc import Sequence as ABCSequence

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer
from tsfm_handle import KEEP_IDX_NO_FAN
from forecast import (
    OBS_DIM,
    OBS_DIM_NO_FAN,
    _denormalize_matrix,
    _normalize_matrix,
    build_default_forecast_metadata,
    build_input_from_history,
    get_future_observations,
    load_forecaster,
    target_cols_to_no_fan_order,
)


class FrameSkip(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip

    def step(self, action: np.ndarray):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, _, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, _, info

    def reset(self):
        return self.env.reset()


ACTIVATION_FN_MAP = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
}


class BDQNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        extended_dim: int,
        action_nvec: Sequence[int],
        net_arch: Sequence[int],
        activation_fn: nn.Module,
    ):
        super().__init__()
        self.action_nvec = list(action_nvec)
        self.observation_dim = observation_dim
        self.extended_dim = extended_dim
        if len(self.action_nvec) == 0:
            raise ValueError("Action vector must not be empty")
        
        # V network: only uses current obs
        v_layers = []
        last_v_dim = observation_dim
        for size in net_arch:
            v_layers.append(nn.Linear(last_v_dim, size))
            v_layers.append(activation_fn())
            last_v_dim = size
        self.value_representation = nn.Sequential(*v_layers)
        self.value_head = nn.Linear(last_v_dim, 1)
        
        # A network: uses extended obs (current obs + future features)
        adv_layers = []
        last_adv_dim = extended_dim
        for size in net_arch:
            adv_layers.append(nn.Linear(last_adv_dim, size))
            adv_layers.append(activation_fn())
            last_adv_dim = size
        self.adv_representation = nn.Sequential(*adv_layers)
        self.adv_heads = nn.ModuleList([nn.Linear(last_adv_dim, size) for size in self.action_nvec])

    def forward(self, extended_obs: torch.Tensor) -> torch.Tensor:
        # Split extended_obs into current obs and future features
        current_obs = extended_obs[:, :self.observation_dim]
        
        # V network: only uses current obs
        value_out = self.value_representation(current_obs)
        value = self.value_head(value_out)
        
        # A network: uses extended obs
        adv_out = self.adv_representation(extended_obs)
        advs = torch.stack([head(adv_out) for head in self.adv_heads], dim=1)
        
        q_value = value.unsqueeze(2) + advs - advs.mean(2, keepdim=True)
        return q_value


def serialize_args(args) -> dict:
    """Convert cli args to JSON-safe primitives."""
    sanitized = {}
    for key, value in vars(args).items():
        if isinstance(value, ABCSequence) and not isinstance(value, (str, bytes)):
            sanitized[key] = list(value)
        else:
            sanitized[key] = value
    return sanitized


def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 3
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = True
    upload_model: bool = False
    hf_entity: str = ""

    env_id: str = "SemiPhysBuildingSim-v0"
    total_timesteps: int = 1_000_000 #1_000_000
    learning_rate: float = 2e-3
    num_envs: int = 1
    buffer_size: int = 500000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 1000
    batch_size: int = 64
    train_frequency: int = 256
    gradient_steps: int = 128
    start_e: float = 1.0
    end_e: float = 0.04
    exploration_fraction: float = 0.16
    learning_starts: int = 20_000
    reward_mode: str = "Baseline_OCC_PPD_with_energy" #"Baseline_OCC_PPD_with_energy"
    tradeoff_constant: int = 20.0
    frame_skip: int = 5
    net_arch: Sequence[int] = (512, 512)
    activation_fn: str = "relu"

    fore_step: int = 3
    obs_dim_no_fan: int = 43
    checkpoint_interval: int = 100_000 #100_000
    # ood to summer
    # forecast_checkpoint: Optional[str] = "/home/users/create/smrksun/world_model/diffuser/results/diffusion_wm/DiffusionWM_Results_20260618_2319/checkpoints/global_DiffusionWM_best.pth"
    # ood to winter
    forecast_checkpoint: Optional[
        str] = "/home/users/create/smrksun/world_model/diffuser/results/diffusion_wm/DiffusionWM_Results_20260615_1757/checkpoints/global_DiffusionWM_best.pth"

    forecast_config: Optional[str] = None
    forecast_metadata: Optional[str] = None
    forecast_T: int = 6
    forecast_N: int = 3
    forecast_F_in: int = 50
    forecast_model_type: str = "DiffusionWM"


def _forecast_model_type_is_diffusion(model_type: str) -> bool:
    return str(model_type).strip().lower() in {
        "diffusion",
        "diffusionwm",
        "diffusion_wm",
        "gaussiandiffusion",
    }


def _resolve_train_config_path(
    checkpoint_path: str, explicit_config: Optional[str]
) -> Path:
    if explicit_config:
        config_path = Path(explicit_config)
        if config_path.is_file():
            return config_path
        raise FileNotFoundError(f"--forecast-config 未找到文件: {explicit_config}")

    ckpt_path = Path(checkpoint_path).resolve()
    for candidate in (
        ckpt_path.parent / "train_config.json",
        ckpt_path.parent.parent / "train_config.json",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "未找到 train_config.json：请设置 --forecast-config，或将 train_config.json "
        "放在 checkpoint 同目录或上一级（与 world model 训练脚本产出一致）。"
    )


def _build_forecast_metadata_from_train_config(
    config: dict[str, Any],
    model_type: str,
) -> dict[str, Any]:
    cols = config.get("columns") or config.get("data_config") or {}
    input_cols = cols.get("input_cols") or config.get("model_config", {}).get("input_cols")
    target_cols = cols.get("target_cols") or config.get("model_config", {}).get("target_cols")
    if not input_cols or not target_cols:
        raise ValueError("train_config.json 缺少 input_cols / target_cols")

    model_config = config.get("model_config") or {}
    T = int(cols.get("T") or model_config.get("T") or 6)
    N = int(cols.get("N") or model_config.get("N") or 3)
    F_in = int(cols.get("F_in") or model_config.get("F_in") or (len(input_cols) // max(T, 1)))
    return {
        "input_cols": [str(c) for c in input_cols],
        "target_cols": [str(c) for c in target_cols],
        "T": T,
        "N": N,
        "F_in": F_in,
        "MODEL_TYPE": model_type,
    }


def _resolve_forecast_metadata(
    args: Args,
    train_config_path: str,
) -> Tuple[Optional[str], Optional[dict[str, Any]]]:
    if args.forecast_metadata and os.path.exists(args.forecast_metadata):
        print(f"Using forecast metadata file: {args.forecast_metadata}")
        return args.forecast_metadata, None

    ckpt_meta = Path(args.forecast_checkpoint).resolve().parent / "forecast_metadata.json"
    if ckpt_meta.is_file():
        print(f"Using checkpoint-side metadata: {ckpt_meta}")
        return str(ckpt_meta), None

    try:
        with open(train_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        forecast_meta = _build_forecast_metadata_from_train_config(
            config, model_type=args.forecast_model_type
        )
        print(
            "Built metadata from train_config: "
            f"T={forecast_meta['T']} N={forecast_meta['N']} F_in={forecast_meta['F_in']}"
        )
        return None, forecast_meta
    except Exception as exc:
        print(f"Could not build metadata from train_config: {exc}")

    forecast_meta = build_default_forecast_metadata(
        T=args.forecast_T,
        N=args.forecast_N,
        F_in=args.forecast_F_in,
        MODEL_TYPE=args.forecast_model_type,
    )
    print(
        "Using default metadata: "
        f"T={args.forecast_T} N={args.forecast_N} F_in={args.forecast_F_in}"
    )
    return None, forecast_meta


def _minmax_norm_flat(x: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mins = np.asarray(mins, dtype=np.float32).reshape(1, -1)
    maxs = np.asarray(maxs, dtype=np.float32).reshape(1, -1)
    d = np.clip(maxs - mins, 1e-6, None)
    return ((x - mins) / d).astype(np.float32)


def _minmax_denorm_flat(x: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mins = np.asarray(mins, dtype=np.float32).reshape(1, -1)
    maxs = np.asarray(maxs, dtype=np.float32).reshape(1, -1)
    d = np.clip(maxs - mins, 1e-6, None)
    return (x * d + mins).astype(np.float32)


def _column_minmax_from_norm_ranges(
    columns: list[str],
    norm_ranges: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """与 forecast._normalize_matrix 一致：按列名语义从 normalization_ranges 推导 mins/maxs。"""
    outdoor_min = norm_ranges.get("OUTDOOR_MIN", 0.0)
    outdoor_max = norm_ranges.get("OUTDOOR_MAX", 40.0)
    sem_min = norm_ranges.get("SEMANTIC_MIN", {})
    sem_max = norm_ranges.get("SEMANTIC_MAX", {})

    def base_range(base: str) -> tuple[float, float]:
        if base == "outdoor_temp":
            return outdoor_min, outdoor_max
        if base in ("FCU_fan_feedback",):
            return sem_min.get("fan", 0.0), sem_max.get("fan", 3.0)
        if "occupant" in base.lower() or base == "occupant_num":
            return sem_min.get("occupant_num", 0.0), sem_max.get("occupant_num", 5.0)
        if base in sem_min:
            return sem_min[base], sem_max[base]
        return 0.0, 1.0

    mins: list[float] = []
    maxs: list[float] = []
    for col in columns:
        base = re.sub(r"_t\d+$", "", col.replace("y_", "", 1) if col.startswith("y_") else col)
        base = re.sub(r"^room\d+_", "", base)
        lo, hi = base_range(base)
        mins.append(lo)
        maxs.append(hi)
    return np.asarray(mins, dtype=np.float32), np.asarray(maxs, dtype=np.float32)


def _resolve_diffusion_column_normalizers(
    config: dict[str, Any],
    input_cols: list[str],
    target_cols: list[str],
    use_train_normalization: bool,
) -> tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    dict[str, Any],
    bool,
]:
    """
    与 forecast_sin / bdq_fu_new_sin 对齐：
    - 优先使用 train_config.normalization.input/target_normalizer（DataNormalizer）
    - 缺失时回退到 normalization_ranges 按列推导 mins/maxs
    - use_train_normalization=False 时仅保留 norm_ranges，前向走 _normalize_matrix
    """
    norm_ranges = config.get("normalization_ranges") or {}
    use_data_norm = bool(use_train_normalization)
    in_mins = in_maxs = tg_mins = tg_maxs = None

    if use_data_norm:
        norm_sec = config.get("normalization") or {}
        inp = norm_sec.get("input_normalizer") or {}
        tgt = norm_sec.get("target_normalizer") or {}
        if inp.get("mins") is not None and inp.get("maxs") is not None:
            in_mins = np.asarray(inp["mins"], dtype=np.float32)
            in_maxs = np.asarray(inp["maxs"], dtype=np.float32)
        if tgt.get("mins") is not None and tgt.get("maxs") is not None:
            tg_mins = np.asarray(tgt["mins"], dtype=np.float32)
            tg_maxs = np.asarray(tgt["maxs"], dtype=np.float32)

        if in_mins is None or in_maxs is None or in_mins.size != len(input_cols):
            in_mins, in_maxs = _column_minmax_from_norm_ranges(input_cols, norm_ranges)
            print(
                "Warning: train_config 缺少 input_normalizer，"
                f"改用 normalization_ranges 推导 input mins/maxs（{len(input_cols)} 列）"
            )
        if tg_mins is None or tg_maxs is None or tg_mins.size != len(target_cols):
            tg_mins, tg_maxs = _column_minmax_from_norm_ranges(target_cols, norm_ranges)
            print(
                "Warning: train_config 缺少 target_normalizer，"
                f"改用 normalization_ranges 推导 target mins/maxs（{len(target_cols)} 列）"
            )

    return in_mins, in_maxs, tg_mins, tg_maxs, norm_ranges, use_data_norm


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _ensure_diffuser_importable() -> None:
    """Load diffuser from cleanrl-master/diffuser/."""
    root = str(_project_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    local_diffuser = _project_root() / "diffuser" / "models"
    if not local_diffuser.is_dir():
        raise ImportError(
            f"未找到本地 diffuser 包: {local_diffuser}。"
            "请将 diffuser 目录放在 cleanrl-master/ 下。"
        )
    import diffuser.models  # noqa: F401


def _load_diffusion_state_dict(module: nn.Module, state: Any, *, strict: bool = True) -> None:
    """
    加载 diffusion 权重。训练时若启用 RC physics loss，checkpoint 会含 physics_loss.*；
    在线推理/采样不需要该模块，加载前剔除这些键即可。
    """
    if not isinstance(state, dict):
        module.load_state_dict(state, strict=strict)
        return
    filtered = {k: v for k, v in state.items() if not k.startswith("physics_loss.")}
    dropped = len(state) - len(filtered)
    if dropped:
        print(
            f"加载 diffusion 权重：忽略 {dropped} 个 physics_loss 参数"
            "（推理阶段不使用 RC loss 模块）"
        )
    module.load_state_dict(filtered, strict=strict)


class DiffusionSemiPhysForecaster:
    """SemiPhys Diffusion world model wrapper (same pattern as DiffusionSinergymForecaster)."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion: Optional[nn.Module] = None
        self.T: int = 0
        self.N: int = 0
        self.F_in: int = 0
        self.input_cols: Optional[list[str]] = None
        self.target_cols: Optional[list[str]] = None
        self.model_type = "DiffusionWM"
        self._history: Optional[np.ndarray] = None
        self._train_input_mins: Optional[np.ndarray] = None
        self._train_input_maxs: Optional[np.ndarray] = None
        self._train_target_mins: Optional[np.ndarray] = None
        self._train_target_maxs: Optional[np.ndarray] = None
        self.norm_ranges: dict[str, Any] = {}
        self._use_train_normalization = True
        self.target_to_no_fan: Optional[list[tuple[int, int]]] = None
        self.sample_batch_size = 1
        self.deterministic_z = True
        self.use_latent_encoder = False
        self.z_dim = 1
        self.constant_z_value = 1.0

    @staticmethod
    def _metadata_matches_normalizer(
        meta: dict[str, Any],
        input_len: Optional[int],
        target_len: Optional[int],
    ) -> bool:
        if input_len is not None and len(meta.get("input_cols") or []) != input_len:
            return False
        if target_len is not None and len(meta.get("target_cols") or []) != target_len:
            return False
        return bool(meta.get("input_cols")) and bool(meta.get("target_cols"))

    def _resolve_meta_dict(
        self,
        checkpoint_path: Path,
        forecast_metadata_path: Optional[str | Path],
        forecast_metadata: Optional[dict[str, Any]],
        train_config: dict[str, Any],
        input_norm_len: Optional[int] = None,
        target_norm_len: Optional[int] = None,
    ) -> dict[str, Any]:
        try:
            train_meta = _build_forecast_metadata_from_train_config(train_config, self.model_type)
        except ValueError:
            train_meta = None

        data_config = train_config.get("data_config") or {}
        if (
            train_meta is not None
            and isinstance(data_config.get("input_cols"), list)
            and isinstance(data_config.get("target_cols"), list)
        ):
            return train_meta

        def pick_meta(source: str, meta: dict[str, Any]) -> Optional[dict[str, Any]]:
            if self._metadata_matches_normalizer(meta, input_norm_len, target_norm_len):
                return meta
            print(
                "Warning: "
                f"{source} 与 train_config normalization 列数不一致 "
                f"(input {len(meta.get('input_cols') or [])} vs {input_norm_len}, "
                f"target {len(meta.get('target_cols') or [])} vs {target_norm_len})，"
                "改用 train_config.json 中的列定义"
            )
            return None

        if forecast_metadata is not None:
            picked = pick_meta("forecast_metadata", forecast_metadata)
            if picked is not None:
                return picked
            if train_meta is not None:
                return train_meta
            return forecast_metadata

        candidates: list[Path] = []
        if forecast_metadata_path:
            candidates.append(Path(forecast_metadata_path))
        ck_dir = checkpoint_path.parent
        candidates.extend(
            [
                ck_dir / "forecast_metadata.json",
                ck_dir.parent / "forecast_metadata.json",
            ]
        )
        for meta_path in candidates:
            if not meta_path.is_file():
                continue
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "input_cols" not in data or "target_cols" not in data:
                continue
            picked = pick_meta(str(meta_path), data)
            if picked is not None:
                return picked
        if train_meta is not None:
            return train_meta
        raise ValueError(
            "未找到可用的 forecast metadata：train_config.json 缺少 input_cols/target_cols，"
            "且 checkpoint 旁无匹配的 forecast_metadata.json"
        )

    def load(
        self,
        checkpoint_path: str | Path,
        config_path: Optional[str | Path] = None,
        forecast_metadata_path: Optional[str | Path] = None,
        forecast_metadata: Optional[dict[str, Any]] = None,
        use_train_normalization: bool = True,
    ) -> None:
        _ensure_diffuser_importable()

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        config_path = (
            Path(config_path)
            if config_path
            else (ckpt_path.parent.parent / "train_config.json")
        )
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config not found: {config_path}. 请提供 diffusion train_config.json"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        meta = self._resolve_meta_dict(
            ckpt_path,
            forecast_metadata_path,
            forecast_metadata,
            config,
            input_norm_len=None,
            target_norm_len=None,
        )
        self.input_cols = [str(c) for c in meta["input_cols"]]
        self.target_cols = [str(c) for c in meta["target_cols"]]
        self.T = int(meta["T"])
        self.N = int(meta["N"])
        self.F_in = int(meta["F_in"])
        self.target_to_no_fan = target_cols_to_no_fan_order(self.target_cols)

        (
            self._train_input_mins,
            self._train_input_maxs,
            self._train_target_mins,
            self._train_target_maxs,
            self.norm_ranges,
            self._use_train_normalization,
        ) = _resolve_diffusion_column_normalizers(
            config,
            self.input_cols,
            self.target_cols,
            use_train_normalization,
        )

        args = self._diffusion_args_from_config(config)
        action_dim = len(self.target_cols) // int(args.horizon)
        observation_dim = len(self.input_cols)
        self.diffusion = self._build_diffusion(action_dim, observation_dim, args)
        payload = torch.load(ckpt_path, map_location="cpu")
        state = (payload.get("ema") or payload.get("model")) if isinstance(payload, dict) else payload
        if state is None:
            state = payload
        _load_diffusion_state_dict(self.diffusion, state, strict=True)
        self.diffusion.to(self.device)
        self.diffusion.eval()

    def _diffusion_args_from_config(self, config: dict[str, Any]) -> SimpleNamespace:
        diffusion_config = config.get("diffusion_config") or {}
        train_config = config.get("train_config") or {}
        all_args = config.get("all_args") or {}

        def get_value(arg_key: str, config_key: str, default: Any) -> Any:
            if arg_key in all_args:
                return all_args[arg_key]
            return diffusion_config.get(config_key, train_config.get(config_key, default))

        wm_horizon_mode = get_value("wm_horizon_mode", "WM_HORIZON_MODE", "flat")
        horizon = int(get_value("horizon", "HORIZON", 1))
        if wm_horizon_mode == "temporal":
            horizon = self.N
        dim_mults = get_value("dim_mults", "DIM_MULTS", (8,))
        self.sample_batch_size = int(get_value("sample_batch_size", "SAMPLE_BATCH_SIZE", 1))
        self.deterministic_z = bool(get_value("deterministic_z", "DETERMINISTIC_Z", True))
        self.use_latent_encoder = bool(get_value("use_latent_encoder", "USE_LATENT_ENCODER", False))
        if self.use_latent_encoder:
            raise NotImplementedError(
                "DiffusionSemiPhysForecaster does not support latent encoder checkpoints."
            )
        self.z_dim = int(get_value("z_dim", "Z_DIM", 1))
        self.constant_z_value = float(get_value("constant_z_value", "CONSTANT_Z_VALUE", 1.0))
        return SimpleNamespace(
            horizon=horizon,
            dim_mults=tuple(dim_mults),
            attention=bool(get_value("attention", "ATTENTION", False)),
            device=self.device,
            n_diffusion_steps=int(get_value("n_diffusion_steps", "N_DIFFUSION_STEPS", 20)),
            loss_type=get_value("loss_type", "LOSS_TYPE", "l1"),
            clip_denoised=bool(get_value("clip_denoised", "CLIP_DENOISED", False)),
            predict_epsilon=bool(get_value("predict_epsilon", "PREDICT_EPSILON", False)),
            action_weight=float(get_value("action_weight", "ACTION_WEIGHT", 10)),
            loss_discount=float(get_value("loss_discount", "LOSS_DISCOUNT", 1.0)),
            action_loss_weights=None,
            smoothl1_beta=float(get_value("smoothl1_beta", "SMOOTHL1_BETA", 0.05)),
            z_dim=self.z_dim,
        )

    def _build_diffusion(self, action_dim: int, observation_dim: int, args: SimpleNamespace):
        if hasattr(torch.backends, "mkldnn"):
            torch.backends.mkldnn.enabled = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
        import diffuser.models as models
        import diffuser.utils as utils

        model_config = utils.Config(
            models.TemporalUnet,
            horizon=args.horizon,
            transition_dim=action_dim + observation_dim + args.z_dim,
            cond_dim=observation_dim + args.z_dim,
            dim_mults=args.dim_mults,
            attention=args.attention,
            device=args.device,
        )
        diffusion_config = utils.Config(
            models.GaussianDiffusion,
            horizon=args.horizon,
            observation_dim=observation_dim + args.z_dim,
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
        return diffusion_config(model_config())

    def set_history(self, history: Optional[np.ndarray]) -> None:
        if history is None:
            self._history = None
            return
        h = np.asarray(history, dtype=np.float32)
        self._history = h.reshape(1, -1) if h.ndim == 1 else h

    def update_history(self, obs: np.ndarray) -> None:
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if self._history is None:
            self._history = np.repeat(obs, max(1, self.T), axis=0)
        else:
            self._history = np.concatenate([self._history, obs], axis=0)[-self.T :]
        if self._history.shape[0] < self.T:
            pad = np.repeat(self._history[:1], self.T - self._history.shape[0], axis=0)
            self._history = np.concatenate([pad, self._history], axis=0)

    def reset_episode(self) -> None:
        self._history = None

    def _condition_from_history(self) -> np.ndarray:
        if self._history is None or self.input_cols is None:
            raise RuntimeError("DiffusionSemiPhysForecaster._history 未设置")
        seq = build_input_from_history(self._history, self.input_cols, self.T, self.F_in)
        cond = seq.reshape(1, -1).astype(np.float32)
        if self._use_train_normalization:
            assert self._train_input_mins is not None and self._train_input_maxs is not None
            return _minmax_norm_flat(cond, self._train_input_mins, self._train_input_maxs)
        assert self.input_cols is not None
        return _normalize_matrix(cond, self.input_cols, self.norm_ranges)

    @torch.no_grad()
    def _sample_denormalized(self, cond_np: np.ndarray) -> np.ndarray:
        if self.diffusion is None:
            raise RuntimeError("DiffusionSemiPhysForecaster not loaded")
        self.diffusion.eval()
        self.diffusion.clip_denoised = True
        outs = []
        for i in range(0, cond_np.shape[0], max(1, self.sample_batch_size)):
            cond = torch.as_tensor(
                cond_np[i : i + self.sample_batch_size],
                dtype=torch.float32,
                device=self.device,
            )
            z = torch.full(
                (cond.shape[0], self.z_dim),
                self.constant_z_value,
                dtype=cond.dtype,
                device=cond.device,
            )
            cond_cat = torch.cat([cond, z], dim=-1)
            cond_dict = {t: cond_cat for t in range(self.diffusion.horizon)}
            sample = self.diffusion.conditional_sample(
                cond_dict,
                horizon=self.diffusion.horizon,
                verbose=False,
            )
            pred = sample.trajectories[:, :, : self.diffusion.action_dim].reshape(cond.shape[0], -1)
            outs.append(pred.detach().cpu().numpy())
        y = np.concatenate(outs, axis=0)
        if self._use_train_normalization:
            assert self._train_target_mins is not None and self._train_target_maxs is not None
            return _minmax_denorm_flat(y, self._train_target_mins, self._train_target_maxs)
        assert self.target_cols is not None
        return _denormalize_matrix(y, self.target_cols, self.norm_ranges)

    def predict_future_no_fan(self, current_obs: np.ndarray) -> np.ndarray:
        current_obs = np.atleast_1d(np.asarray(current_obs, dtype=np.float32)).ravel()
        if self._history is None:
            self.set_history(np.repeat(current_obs.reshape(1, -1), max(1, self.T), axis=0))
        else:
            self.update_history(current_obs)

        y_phys = self._sample_denormalized(self._condition_from_history())
        out = np.zeros((self.N, OBS_DIM_NO_FAN), dtype=np.float32)
        n = y_phys.shape[1] if y_phys.ndim == 2 else y_phys.shape[0]
        assert self.target_to_no_fan is not None
        for i in range(min(n, len(self.target_to_no_fan))):
            k, no_fan_idx = self.target_to_no_fan[i]
            step_idx = k - 1
            if 0 <= step_idx < self.N and 0 <= no_fan_idx < out.shape[1]:
                val = y_phys[0, i] if y_phys.ndim == 2 else y_phys[i]
                out[step_idx, no_fan_idx] = val
        return out

    def get_future_observations(
        self, current_obs: np.ndarray, action: Any, fore_step: int
    ) -> np.ndarray:
        del action
        pred_no_fan = self.predict_future_no_fan(current_obs)
        if fore_step > self.N:
            last = pred_no_fan[-1:]
            extra = np.repeat(last, fore_step - self.N, axis=0)
            pred_no_fan = np.concatenate([pred_no_fan, extra], axis=0)
        pred_no_fan = pred_no_fan[:fore_step]

        current_obs = np.atleast_1d(np.asarray(current_obs, dtype=np.float32)).ravel()
        fan_indices = [1 + r * 7 + 1 for r in range(7)]
        future_50 = np.zeros((fore_step, OBS_DIM), dtype=np.float32)
        future_50[:, KEEP_IDX_NO_FAN] = pred_no_fan
        for idx in fan_indices:
            future_50[:, idx] = current_obs[idx]
        return future_50


def load_diffusion_spbs_forecaster(
    checkpoint_path: str | Path,
    config_path: Optional[str | Path] = None,
    forecast_metadata_path: Optional[str | Path] = None,
    forecast_metadata: Optional[dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    use_train_normalization: bool = True,
) -> DiffusionSemiPhysForecaster:
    forecaster = DiffusionSemiPhysForecaster(device=device)
    forecaster.load(
        checkpoint_path,
        config_path=config_path,
        forecast_metadata_path=forecast_metadata_path,
        forecast_metadata=forecast_metadata,
        use_train_normalization=use_train_normalization,
    )
    return forecaster


def load_world_model_forecaster(args: Args, device: torch.device):
    """Load DiffusionWM or MambaFormer/LSTM forecaster (same flow as bdq_fu_new_sin)."""
    if not args.forecast_checkpoint:
        raise ValueError("forecast_checkpoint is required to use forecast model.")

    train_config_path = str(
        _resolve_train_config_path(args.forecast_checkpoint, args.forecast_config)
    )

    if _forecast_model_type_is_diffusion(args.forecast_model_type):
        print(f"Loading DiffusionWM from {args.forecast_checkpoint}")
        print(f"Using train_config: {train_config_path}")
        meta_file, forecast_meta = _resolve_forecast_metadata(args, train_config_path)
        forecaster = load_diffusion_spbs_forecaster(
            args.forecast_checkpoint,
            config_path=train_config_path,
            forecast_metadata_path=meta_file,
            forecast_metadata=forecast_meta if meta_file is None else None,
            device=device,
            use_train_normalization=True,
        )
        print(
            f"DiffusionWM loaded: T={forecaster.T}, N={forecaster.N}, F_in={forecaster.F_in}"
        )
        return forecaster, train_config_path

    meta_file, forecast_meta = _resolve_forecast_metadata(args, train_config_path)
    forecaster = load_forecaster(
        args.forecast_checkpoint,
        config_path=train_config_path,
        forecast_metadata_path=meta_file,
        forecast_metadata=forecast_meta if meta_file is None else None,
        device=device,
    )
    print(
        f"Forecaster loaded ({args.forecast_model_type}): "
        f"T={forecaster.T}, N={forecaster.N}, F_in={forecaster.F_in}"
    )
    return forecaster, train_config_path


def predict_future_with_history(
    forecaster,
    history: np.ndarray,
    current_obs: np.ndarray,
    action,
    steps: int,
) -> np.ndarray:
    if steps <= 0:
        return np.empty((0, current_obs.shape[0]), dtype=np.float32)
    history_backup = None if getattr(forecaster, "_history", None) is None else forecaster._history.copy()
    try:
        forecaster.set_history(np.asarray(history, dtype=np.float32))
        if hasattr(forecaster, "get_future_observations"):
            return forecaster.get_future_observations(current_obs, action, steps)
        return get_future_observations(forecaster, current_obs, action, steps)
    finally:
        forecaster._history = history_backup


def reset_forecaster_state(forecaster) -> None:
    if hasattr(forecaster, "reset_episode"):
        forecaster.reset_episode()
    else:
        forecaster._history = None


def make_env(env_id, seed, idx, capture_video, run_name, reward_mode, tradeoff_constant, frame_skip):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(
                env_id,
                reward_mode=reward_mode,
                tradeoff_constant=tradeoff_constant,
                eval_mode=True,
                USE_Multi_Discrete=True,
            )
            env = FrameSkip(env, skip=frame_skip)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def to_python_action(action):
    if isinstance(action, np.ndarray):
        if action.ndim == 0:
            return int(action)
        if action.ndim == 1 and action.shape[0] == 1:
            return int(action[0])
        return action.tolist()
    if isinstance(action, (np.generic,)):
        return int(action)
    if isinstance(action, (list, tuple)):
        return list(action)
    return action


def zero_action(action_space):
    if isinstance(action_space, gym.spaces.Discrete):
        return 0
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        return [0] * len(action_space.nvec)
    return action_space.sample()


def flatten_future_obs(future_obs: np.ndarray, args) -> np.ndarray:
    if args.fore_step <= 0:
        return np.zeros(0, dtype=np.float32)
    if future_obs.shape[0] == 0:
        return np.zeros(args.fore_step * args.obs_dim_no_fan, dtype=np.float32)

    to_float = np.asarray(future_obs, dtype=np.float32)
    no_fan = to_float[:, KEEP_IDX_NO_FAN]
    if no_fan.shape[0] < args.fore_step:
        if no_fan.shape[0] == 0:
            padding = np.zeros((args.fore_step, args.obs_dim_no_fan), dtype=np.float32)
            no_fan = padding
        else:
            last = no_fan[-1:]
            pad = np.repeat(last, args.fore_step - no_fan.shape[0], axis=0)
            no_fan = np.concatenate([no_fan, pad], axis=0)
    return no_fan[: args.fore_step].reshape(-1)


def polyak_update(source: nn.Module, target: nn.Module, tau: float):
    for src, tgt in zip(source.parameters(), target.parameters()):
        tgt.data.copy_(tau * src.data + (1.0 - tau) * tgt.data)


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = (
        f"bdq_wm_new_ts/{args.reward_mode}/tradeoff_{args.tradeoff_constant}/seed_{args.seed}_skip_{args.frame_skip}_"
        f"{time.strftime('%Y%m%d_%H%M', time.localtime())}"
    )

    if args.track:
        import swanlab

        swanlab.init(
            project=args.wandb_project_name,
            team=args.wandb_entity,
            config=vars(args),
            name=run_name,
            save_code=True,
            monitor=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed + i,
                i,
                args.capture_video,
                run_name,
                args.reward_mode,
                args.tradeoff_constant,
                args.frame_skip,
            )
            for i in range(args.num_envs)
        ]
    )

    if isinstance(envs.single_action_space, gym.spaces.Discrete):
        action_nvec = [envs.single_action_space.n]
    elif isinstance(envs.single_action_space, gym.spaces.MultiDiscrete):
        action_nvec = envs.single_action_space.nvec.tolist()
    else:
        raise NotImplementedError("Only Discrete/MultiDiscrete actions supported")

    activation_cls = ACTIVATION_FN_MAP.get(args.activation_fn.lower(), nn.ReLU)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    extended_dim = obs_dim + args.fore_step * args.obs_dim_no_fan

    extended_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(extended_dim,), dtype=np.float32
    )

    rb = ReplayBuffer(
        args.buffer_size,
        extended_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    model_dir = f"runs/{run_name}"
    os.makedirs(model_dir, exist_ok=True)
    params_path = os.path.join(model_dir, "params.json")
    with open(params_path, "w", encoding="utf-8") as params_file:
        json.dump(serialize_args(args), params_file, indent=2, ensure_ascii=False)
    checkpoint_interval = args.checkpoint_interval

    q_network = BDQNetwork(obs_dim, extended_dim, action_nvec, args.net_arch, activation_cls).to(device)
    target_network = BDQNetwork(obs_dim, extended_dim, action_nvec, args.net_arch, activation_cls).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

    gradient_updates = 0
    start_time = time.time()

    obs, _ = envs.reset()
    obs = obs[0].astype(np.float32)
    forecaster, train_config_path = load_world_model_forecaster(args, device)
    if args.fore_step > forecaster.N:
        raise ValueError(
            f"--fore-step={args.fore_step} 大于模型预测步数 N={forecaster.N}，"
            "请减小 fore_step 或更换 checkpoint。"
        )

    history_len = int(forecaster.T)
    obs_history: deque[np.ndarray] = deque(maxlen=history_len)
    for _ in range(history_len):
        obs_history.append(obs.copy())

    prev_action = zero_action(envs.single_action_space)
    history_pre = np.stack(list(obs_history), axis=0)
    future_feature = flatten_future_obs(
        predict_future_with_history(
            forecaster, history_pre, obs, prev_action, args.fore_step
        ),
        args,
    )
    extended_obs = np.concatenate([obs, future_feature], axis=-1)
    episode_return = 0.0
    best_episodic_return = float("-inf")
    ckpt_metadata = {
        "obs_dim": obs_dim,
        "extended_dim": extended_dim,
        "fore_step": args.fore_step,
        "obs_dim_no_fan": args.obs_dim_no_fan,
        "action_nvec": action_nvec,
        "use_world_model": True,
        "forecast_model_type": args.forecast_model_type,
        "forecast_checkpoint": args.forecast_checkpoint,
        "forecast_config": args.forecast_config,
        "train_config_resolved": train_config_path,
        "world_model_T": forecaster.T,
        "world_model_N": forecaster.N,
    }

    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )

        if random.random() < epsilon:
            selected_action = envs.single_action_space.sample()
        else:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(
                    extended_obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                q_values = q_network(obs_tensor)
                greedy_actions = torch.argmax(q_values, dim=2).cpu().numpy()
                selected = greedy_actions[0]
                if selected.shape == ():
                    selected_action = int(selected)
                elif selected.ndim == 1 and selected.shape[0] == 1:
                    selected_action = int(selected[0])
                else:
                    selected_action = selected.tolist()

        action_python = to_python_action(selected_action)
        action_array = np.array(action_python, dtype=np.int64)
        if action_array.ndim == 0:
            action_array = action_array.reshape(1)
        batched_action = np.expand_dims(action_array, 0)
        history_pre = np.stack(list(obs_history), axis=0)

        next_obs, rewards, dones, _, infos = envs.step(batched_action)
        next_obs = next_obs[0].astype(np.float32)
        reward = float(rewards[0])
        done = bool(dones[0])

        future_feature_next = flatten_future_obs(
            predict_future_with_history(
                forecaster, history_pre, next_obs, action_python, args.fore_step
            ),
            args,
        )
        extended_next_obs = np.concatenate([next_obs, future_feature_next], axis=-1)

        infos_list = [infos[0]] if isinstance(infos, list) else [infos]
        rb.add(
            extended_obs,
            extended_next_obs,
            np.array(action_array, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.array(done, dtype=np.float32),
            infos_list,
        )

        extended_obs = extended_next_obs
        episode_return += reward
        if done:
            reset_forecaster_state(forecaster)
            obs_history.clear()
            for _ in range(history_len):
                obs_history.append(next_obs.copy())
            prev_action = zero_action(envs.single_action_space)
            reset_history_pre = np.stack(list(obs_history), axis=0)
            reset_future_feature = flatten_future_obs(
                predict_future_with_history(
                    forecaster, reset_history_pre, next_obs, prev_action, args.fore_step
                ),
                args,
            )
            extended_obs = np.concatenate([next_obs, reset_future_feature], axis=-1)
            writer.add_scalar("charts/episodic_return", episode_return, global_step)
            if args.track:
                import swanlab

                swanlab.log({"charts/episodic_return": episode_return}, step=global_step)
            if args.save_model and episode_return > best_episodic_return:
                best_episodic_return = episode_return
                best_model_path = f"{model_dir}/best_model.pt"
                torch.save(
                    {
                        "q_network": q_network.state_dict(),
                        "target_network": target_network.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_episodic_return": best_episodic_return,
                        "metadata": ckpt_metadata,
                    },
                    best_model_path,
                )
                print(
                    f"Best model saved to {best_model_path} (episodic_return={episode_return:.2f})"
                )
            else:
                best_episodic_return = max(best_episodic_return, episode_return)
            episode_return = 0.0
        else:
            obs_history.append(next_obs.copy())

        if (
            global_step > args.learning_starts
            and rb.size() >= args.batch_size
            and global_step % args.train_frequency == 0
        ):
            for _ in range(args.gradient_steps):
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    next_q_values = q_network(data.next_observations)
                    next_actions = torch.argmax(next_q_values, dim=2)
                    target_next_q = target_network(data.next_observations)
                    max_next_q = target_next_q.gather(2, next_actions.unsqueeze(2)).squeeze(-1)
                    max_next_q = max_next_q.mean(1, keepdim=True)
                    td_target = data.rewards + args.gamma * (1 - data.dones) * max_next_q

                current_q_values = q_network(data.observations)
                actions_tensor = data.actions.long().reshape(
                    data.observations.shape[0], -1, 1
                )
                current_q_values = current_q_values.gather(2, actions_tensor).squeeze(-1)
                current_q_values = current_q_values.mean(1, keepdim=True)

                loss = F.smooth_l1_loss(current_q_values, td_target)
                if gradient_updates % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss.item(), gradient_updates)
                    writer.add_scalar(
                        "losses/q_values", current_q_values.mean().item(), gradient_updates
                    )
                    writer.add_scalar(
                        "charts/SPS",
                        int((gradient_updates + 1) / max(1, time.time() - start_time)),
                        gradient_updates,
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                gradient_updates += 1
                if gradient_updates % args.target_network_frequency == 0:
                    polyak_update(q_network, target_network, args.tau)

        if (
            args.save_model
            and global_step > 0
            and checkpoint_interval > 0
            and global_step % checkpoint_interval == 0
        ):
            checkpoint_path = f"{model_dir}/checkpoint_step_{global_step}.pt"
            torch.save(
                {
                    "q_network": q_network.state_dict(),
                    "target_network": target_network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "best_episodic_return": best_episodic_return,
                    "metadata": ckpt_metadata,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved to {checkpoint_path} (step={global_step})")

    if args.save_model:
        final_model_path = f"{model_dir}/final_model.pt"
        torch.save(
            {
                "q_network": q_network.state_dict(),
                "target_network": target_network.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": args.total_timesteps,
                "best_episodic_return": best_episodic_return,
                "metadata": ckpt_metadata,
            },
            final_model_path,
        )
        print(f"Final model saved to {final_model_path}")

    envs.close()
    writer.close()
