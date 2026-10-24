"""Shared configuration loader for all lambda_barre entry points.

Reads ``config.toml`` from the repository root and provides helpers to merge
it with built-in defaults and CLI arguments.

Precedence: CLI arguments > config.toml > built-in defaults.
"""
from __future__ import annotations

import argparse
import os
import tomllib

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_REPO_ROOT, "config.toml")


def load_config(path: str = _CONFIG_PATH) -> dict:
    """Load config.toml and return a flat dict of parameter overrides.

    Nested TOML sections are flattened with ``_`` separators so they can
    be merged with argparse namespaces. Returns an empty dict if the file
    does not exist (all defaults apply).
    """
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    flat: dict = {}
    for section, values in raw.items():
        if isinstance(values, dict):
            for key, val in values.items():
                flat[f"{section}_{key}"] = val
        else:
            flat[section] = values
    return flat


# Maps flattened TOML keys (section_key) to the canonical parameter name
# used across all entry points.
TOML_TO_PARAM: dict[str, str] = {
    # transformer section
    "transformer_d_model":   "d_model",
    "transformer_nhead":     "nhead",
    "transformer_layers":    "wm_layers",
    "transformer_dim_ff":    "dim_ff",
    "transformer_dropout":   "wm_dropout",
    # policy section
    "policy_hidden":         "pol_hidden",
    # training section
    "training_lr_wm":          "lr_wm",
    "training_lr_pol":         "lr_pol",
    "training_act_std":        "act_std",
    "training_explore_std":    "explore_std",
    "training_wm_epochs":      "wm_epochs",
    "training_wm_batch":       "wm_batch",
    "training_wm_target_loss": "wm_target_loss",
    "training_pol_steps":      "pol_steps",
    # buffer section
    "buffer_capacity":         "buffer_capacity",
    "buffer_pool_capacity":    "buffer_pool_capacity",
    # sequence section
    "sequence_seq_steps": "seq_steps",
    "sequence_n_ctx":     "n_ctx",
    "sequence_wm_hz":     "wm_hz",
    "sequence_pol_hz":    "pol_hz",
    # runtime section
    "runtime_device":            "device",
    "runtime_accel":             "accel",
    "runtime_seed":              "seed",
    "runtime_compile_wm":        "compile_wm",
    "runtime_use_kv_cache":      "use_kv_cache",
    "runtime_pol_eos_lookahead": "pol_eos_lookahead",
    "runtime_freeze_physics":    "freeze_physics",
    "runtime_no_smooth":         "no_smooth",
}


def apply_toml(params: dict, cfg: dict) -> dict:
    """Merge flattened TOML config into a params dict (mutates and returns it)."""
    for toml_key, param_key in TOML_TO_PARAM.items():
        if toml_key in cfg:
            params[param_key] = cfg[toml_key]
    return params


def merge_cli(params: dict, cli: dict) -> dict:
    """Merge CLI arguments (from argparse with SUPPRESS defaults) into params.

    Handles negated flags (``no_kv_cache`` → ``use_kv_cache = False``,
    ``no_eos_lookahead`` → ``pol_eos_lookahead = False``).
    """
    for key, val in cli.items():
        if key == "no_kv_cache":
            params["use_kv_cache"] = not val
        elif key == "no_eos_lookahead":
            params["pol_eos_lookahead"] = not val
        else:
            params[key] = val
    return params
