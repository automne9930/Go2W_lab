# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export an RSL-RL actor checkpoint to TorchScript (.pt) and ONNX (.onnx).

This is the standalone counterpart of the export block in ``play.py``. Unlike
``play.py`` it does **not** launch Isaac Sim, create an environment or touch the
GPU: it rebuilds the actor network from ``actor_state_dict`` and exports it on
CPU. It can therefore be run at any time, including while a training job is
using the same GPU, without disturbing that job.

The network shape (observation/action dimensions, hidden layers, observation
normalizer) is inferred from the checkpoint itself. The activation function and
the output distribution are read from the run's ``params/agent.yaml``, which
``train.py`` dumps next to the checkpoints.

Usage:
    python scripts/rsl_rl/export.py logs/rsl_rl/<experiment>/<run>/model_800.pt
    python scripts/rsl_rl/export.py logs/rsl_rl/<experiment>/<run>
    python scripts/rsl_rl/export.py <checkpoint> -o /tmp/exported --filename model_800

If Isaac Lab is not installed in the active Python environment, use
``<path_to_IsaacLab>/isaaclab.sh -p`` instead of ``python``. This script itself
does not need Isaac Sim; any Python environment with rsl-rl-lib, torch, onnx
and pyyaml works.

Without ``-o`` the files are written to ``<checkpoint_dir>/exported/policy.{pt,onnx}``,
matching ``play.py``.
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import time

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel

_LINEAR_WEIGHT_RE = re.compile(r"^mlp\.(\d+)\.weight$")
_CHECKPOINT_FILE_RE = re.compile(r"^model_(\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an RSL-RL checkpoint without starting Isaac Sim.")
    parser.add_argument(
        "checkpoint",
        help="Path to a model_*.pt file, or a run directory (the latest checkpoint in it is used).",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=None,
        help="Directory for the exported files. Defaults to <checkpoint_dir>/exported (same as play.py).",
    )
    parser.add_argument(
        "--filename", default="policy", help="Base name of the exported files (default: policy)."
    )
    parser.add_argument(
        "--activation",
        default=None,
        help="Override the actor activation (normally read from params/agent.yaml, default: elu).",
    )
    parser.add_argument(
        "--agent_yaml",
        default=None,
        help="Path to params/agent.yaml. Defaults to <checkpoint_dir>/params/agent.yaml if it exists.",
    )
    parser.add_argument("--no_jit", action="store_true", help="Skip TorchScript export.")
    parser.add_argument("--no_onnx", action="store_true", help="Skip ONNX export.")
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Retries if the checkpoint cannot be read (e.g. it is being written right now). Default: 5.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose ONNX export.")
    return parser.parse_args()


def _checkpoint_iteration(filename: str) -> int:
    match = _CHECKPOINT_FILE_RE.match(filename)
    return int(match.group(1)) if match else -1


def resolve_checkpoint(path: str) -> str:
    """Return a checkpoint file path; if a directory is given, pick its latest model_*.pt."""
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        candidates = [name for name in os.listdir(path) if _checkpoint_iteration(name) >= 0]
        if not candidates:
            raise FileNotFoundError(f"No model_*.pt checkpoint found in directory: {path}")
        latest = max(candidates, key=_checkpoint_iteration)
        print(f"[INFO] Latest checkpoint in run directory: {latest}")
        return os.path.join(path, latest)
    raise FileNotFoundError(f"Checkpoint file or run directory not found: {path}")


def load_checkpoint(path: str, retries: int) -> dict:
    """Load a checkpoint, retrying in case training is writing it at the same moment."""
    last_error: Exception | None = None
    for attempt in range(1, max(retries, 1) + 1):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:  # noqa: BLE001 - surface the original error after the last attempt
            last_error = error
            if attempt < retries:
                print(f"[WARN] Cannot read checkpoint yet (attempt {attempt}/{retries}): {error}")
                print("[WARN] It may be mid-write by the training process; retrying in 2 s.")
                time.sleep(2.0)
    raise RuntimeError(f"Failed to load checkpoint after {retries} attempts: {path}") from last_error


def load_actor_cfg(checkpoint_path: str, agent_yaml: str | None) -> dict:
    """Read the actor section of params/agent.yaml, if available."""
    if agent_yaml is None:
        candidate = os.path.join(os.path.dirname(checkpoint_path), "params", "agent.yaml")
        agent_yaml = candidate if os.path.isfile(candidate) else None
    if agent_yaml is None:
        print("[WARN] params/agent.yaml not found; falling back to checkpoint inference (activation: elu).")
        return {}
    import yaml

    with open(agent_yaml) as file:
        cfg = yaml.safe_load(file) or {}
    print(f"[INFO] Loaded actor config from: {agent_yaml}")
    return cfg.get("actor", {}) or {}


def infer_actor_cfg(state_dict: dict, actor_cfg: dict, activation_override: str | None) -> dict:
    """Infer the MLPModel constructor arguments from the checkpoint and agent.yaml."""
    linear_layers = {int(match.group(1)): state_dict[key].shape for key in state_dict
                     if (match := _LINEAR_WEIGHT_RE.match(key))}
    if not linear_layers:
        raise ValueError(
            "No 'mlp.*.weight' entries in actor_state_dict. "
            "This checkpoint was likely written by an rsl-rl version older than 5.0; "
            "export it with play.py instead."
        )

    indices = sorted(linear_layers)
    obs_dim = int(linear_layers[indices[0]][1])
    mlp_output_dim = int(linear_layers[indices[-1]][0])
    hidden_dims = [int(linear_layers[index][0]) for index in indices[:-1]]

    # The output distribution decides whether the last MLP layer outputs actions directly
    # or a stack of distribution parameters (e.g. 2 * action_dim for a heteroscedastic Gaussian).
    distribution_cfg = copy.deepcopy(actor_cfg.get("distribution_cfg"))
    if distribution_cfg is None:
        if "distribution.std_param" in state_dict:
            distribution_cfg = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}
        elif "distribution.log_std_param" in state_dict:
            distribution_cfg = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "log"}
    distribution_name = str(distribution_cfg.get("class_name", "")) if distribution_cfg else ""
    if "Heteroscedastic" in distribution_name:
        if mlp_output_dim % 2 != 0:
            raise ValueError(f"Unexpected MLP output dim {mlp_output_dim} for {distribution_name}.")
        action_dim = mlp_output_dim // 2
    else:
        action_dim = mlp_output_dim

    # Observation normalization leaves buffers in the state dict, so it is reliable there.
    checkpoint_has_normalizer = any(key.startswith("obs_normalizer.") for key in state_dict)
    if checkpoint_has_normalizer != bool(actor_cfg.get("obs_normalization", False)):
        print(
            "[WARN] obs_normalization mismatch: agent.yaml says "
            f"{bool(actor_cfg.get('obs_normalization', False))}, checkpoint indicates "
            f"{checkpoint_has_normalizer}. Using the checkpoint."
        )

    yaml_hidden_dims = actor_cfg.get("hidden_dims")
    if yaml_hidden_dims is not None and list(yaml_hidden_dims) != hidden_dims:
        print(f"[WARN] hidden_dims mismatch: agent.yaml says {list(yaml_hidden_dims)}, checkpoint has {hidden_dims}.")

    return {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dims": hidden_dims,
        "activation": activation_override or actor_cfg.get("activation", "elu"),
        "obs_normalization": checkpoint_has_normalizer,
        "distribution_cfg": distribution_cfg,
    }


def build_actor(state_dict: dict, cfg: dict) -> MLPModel:
    """Rebuild the actor and load its weights."""
    # MLPModel only needs the observation groups and their dimensions at construction time.
    dummy_obs = TensorDict({"obs": torch.zeros(1, cfg["obs_dim"])}, batch_size=[1])
    model = MLPModel(
        obs=dummy_obs,
        obs_groups={"actor": ["obs"]},
        obs_set="actor",
        output_dim=cfg["action_dim"],
        hidden_dims=cfg["hidden_dims"],
        activation=cfg["activation"],
        obs_normalization=cfg["obs_normalization"],
        distribution_cfg=copy.deepcopy(cfg["distribution_cfg"]),
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def export_jit(model: MLPModel, save_path: str) -> None:
    """Same export as OnPolicyRunner.export_policy_to_jit."""
    jit_model = model.as_jit().to("cpu")
    traced_model = torch.jit.script(jit_model)
    traced_model.save(save_path)


def export_onnx(model: MLPModel, save_path: str, verbose: bool = False) -> None:
    """Same export as OnPolicyRunner.export_policy_to_onnx."""
    onnx_model = model.as_onnx(verbose=verbose).to("cpu")
    onnx_model.eval()
    torch.onnx.export(
        onnx_model,
        onnx_model.get_dummy_inputs(),
        save_path,
        export_params=True,
        opset_version=18,
        verbose=verbose,
        input_names=onnx_model.input_names,
        output_names=onnx_model.output_names,
    )


def verify_jit(model: MLPModel, jit_path: str, obs_dim: int) -> None:
    """Check that the traced policy reproduces the eager actor on a random batch."""
    observations = torch.randn(4, obs_dim)
    with torch.inference_mode():
        expected = model(TensorDict({"obs": observations}, batch_size=[4]))
        traced_model = torch.jit.load(jit_path, map_location="cpu")
        actual = traced_model(observations)
    max_error = (actual - expected).abs().max().item()
    if not torch.allclose(actual, expected, atol=1e-6):
        print(f"[WARN] TorchScript policy differs from the eager actor (max abs diff: {max_error:.3e}).")
    else:
        print(f"[INFO] TorchScript policy verified (max abs diff: {max_error:.3e}).")


def main() -> None:
    args = parse_args()

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, args.retries)
    if "actor_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint has no 'actor_state_dict' (keys: {list(checkpoint.keys())}). "
            "It was probably written by an rsl-rl version older than 5.0; export it with play.py instead."
        )
    actor_state_dict = checkpoint["actor_state_dict"]

    actor_cfg = load_actor_cfg(checkpoint_path, args.agent_yaml)
    cfg = infer_actor_cfg(actor_state_dict, actor_cfg, args.activation)

    print(f"[INFO] Checkpoint : {checkpoint_path}")
    if "iter" in checkpoint:
        print(f"[INFO] Iteration  : {checkpoint['iter']}")
    print(
        f"[INFO] Actor      : obs_dim={cfg['obs_dim']}, action_dim={cfg['action_dim']}, "
        f"hidden_dims={cfg['hidden_dims']}, activation={cfg['activation']}, "
        f"obs_normalization={cfg['obs_normalization']}, "
        f"distribution={cfg['distribution_cfg'] or 'deterministic'}"
    )

    model = build_actor(actor_state_dict, cfg)

    output_dir = args.output_dir or os.path.join(os.path.dirname(checkpoint_path), "exported")
    os.makedirs(output_dir, exist_ok=True)

    jit_path = os.path.join(output_dir, f"{args.filename}.pt")
    onnx_path = os.path.join(output_dir, f"{args.filename}.onnx")

    if not args.no_jit:
        export_jit(model, jit_path)
        print(f"[INFO] TorchScript: {jit_path}")
        verify_jit(model, jit_path, cfg["obs_dim"])
    if not args.no_onnx:
        export_onnx(model, onnx_path, verbose=args.verbose)
        print(f"[INFO] ONNX       : {onnx_path}")

    print("[INFO] Export finished.")


if __name__ == "__main__":
    main()
