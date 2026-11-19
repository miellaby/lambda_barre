"""Shared configuration loader for all lambda_barre entry points.

Reads ``config.toml`` from the repository root and provides helpers to merge
it with built-in defaults and CLI arguments.

Precedence: CLI arguments > config.toml > built-in defaults.
"""
from __future__ import annotations

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
    "transformer_d_model":         "d_model",
    "transformer_nhead":           "nhead",
    "transformer_layers":          "wm_layers",
    "transformer_dim_ff":          "dim_ff",
    "transformer_dropout":         "wm_dropout",
    # policy section
    "policy_hidden":               "pol_hidden",
    # training section
    "training_lr_wm":              "lr_wm",
    "training_lr_pol":             "lr_pol",
    "training_act_std":            "act_std",
    "training_act_conseq_shift":   "act_conseq_shift",
    "training_wm_coreset_epoch":   "wm_coreset_epoch",
    "training_wm_addendum_epoch":  "wm_addendum_epoch",
    "training_wm_batch":           "wm_batch",
    "training_wm_target_loss":     "wm_target_loss",
    "training_pol_steps":          "pol_steps",
    "training_eos_decay":          "eos_decay",
    "training_eos_n_modes":        "eos_n_modes",
    "training_eos_sigma_min":      "eos_sigma_min",
    "training_eos_nll_w":          "eos_nll_w",
    "training_eos_temperature":    "eos_temperature",
    "training_eos_top_p":          "eos_top_p",
    "training_eos_min_p":          "eos_min_p",
    "training_wm_impact_floor":    "wm_impact_floor",
    "training_wm_input_noise":     "wm_input_noise",
    # buffer section
    "buffer_capacity":             "buffer_capacity",
    "buffer_pool_capacity":        "buffer_pool_capacity",
    "buffer_dedup_eps":            "dedup_eps",
    "buffer_dedup_latent_eps":     "dedup_latent_eps",
    "buffer_vivacity_decay":       "vivacity_decay",
    # sequence section
    "sequence_seq_steps":          "seq_steps",
    "sequence_n_ctx":              "n_ctx",
    "sequence_lookahead":          "lookahead",
    "sequence_wm_hz":              "wm_hz",
    "sequence_pol_hz":             "pol_hz",
    # runtime section
    "runtime_device":              "device",
    "runtime_accel":               "accel",
    "runtime_seed":                "seed",
    "runtime_compile_wm":          "compile_wm",
    "runtime_no_smooth":           "no_smooth",
}


# Canonical Brain-related parameter defaults shared by all entry points.
BRAIN_DEFAULTS: dict = {
    "device": os.environ.get("LAMBDA_DEVICE", "cpu"),
    "accel": False,
    "seed": 0,
    "compile_wm": False,
    "d_model": 64,
    "nhead": 4,
    "wm_layers": 3,
    "dim_ff": 384,
    "wm_dropout": 0.0,
    "pol_hidden": 256,
    "lr_wm": 3e-4,
    "lr_pol": 2e-4,
    "act_std": 0.0,
    "act_conseq_shift": 1,
    "seq_steps": 12,
    "n_ctx": 6,
    "lookahead": 10,
    "wm_hz": 4.0,
    "pol_hz": 8.0,
    "buffer_capacity": 1000,
    "buffer_pool_capacity": 1000,
    "dedup_eps": 0.02,
    "dedup_latent_eps": 0.02,
    "vivacity_decay": 0.95,
    "eos_decay": 0.91,
    "eos_n_modes": 8,
    "eos_sigma_min": 0.02,
    "eos_nll_w": 0.1,
    "eos_temperature": 1.0,
    "eos_top_p": 0.9,
    "eos_min_p": 1e-3,
    "wm_impact_floor": 0.10,
    "wm_input_noise": 0.015,
    "wm_coreset_epoch": 300,
    "wm_addendum_epoch": 300,
    "wm_batch": 64,
    "wm_target_loss": 0.004,
    "pol_steps": 32,
}


# Maps canonical entry-point param keys -> Brain.__init__ keyword argument names.
_PARAM_TO_BRAIN_KWARG: dict[str, str] = {
    "device":               "device",
    "seed":                 "seed",
    "compile_wm":           "compile_wm",
    "seq_steps":            "seq_steps",
    "n_ctx":                "n_ctx",
    "lookahead":            "lookahead",
    "d_model":              "d_model",
    "nhead":                "nhead",
    "wm_layers":            "layers",
    "dim_ff":               "dim_ff",
    "wm_dropout":           "dropout",
    "pol_hidden":           "pol_hidden",
    "lr_wm":                "lr_wm",
    "lr_pol":               "lr_pol",
    "act_std":              "act_std",
    "act_conseq_shift":     "act_conseq_shift",
    "buffer_capacity":      "buffer_capacity",
    "buffer_pool_capacity": "buffer_pool_capacity",
    "dedup_eps":            "dedup_eps",
    "dedup_latent_eps":     "dedup_latent_eps",
    "vivacity_decay":       "vivacity_decay",
    "eos_decay":            "eos_decay",
    "eos_n_modes":          "eos_n_modes",
    "eos_sigma_min":        "eos_sigma_min",
    "eos_nll_w":            "eos_nll_w",
    "eos_temperature":      "eos_temperature",
    "eos_top_p":            "eos_top_p",
    "eos_min_p":            "eos_min_p",
    "wm_impact_floor":      "wm_impact_floor",
    "wm_input_noise":       "wm_input_noise",
}


def brain_kwargs(params: dict) -> dict:
    """Extract a clean kwargs dict for ``Brain(...)`` from a merged params dict."""
    return {
        brain_key: params[param_key]
        for param_key, brain_key in _PARAM_TO_BRAIN_KWARG.items()
        if param_key in params
    }


def apply_toml(params: dict, cfg: dict) -> dict:
    """Merge flattened TOML config into a params dict (mutates and returns it)."""
    for toml_key, param_key in TOML_TO_PARAM.items():
        if toml_key in cfg:
            params[param_key] = cfg[toml_key]
    return params


def merge_cli(params: dict, cli: dict) -> dict:
    """Merge CLI arguments (from argparse with SUPPRESS defaults) into params."""
    for key, val in cli.items():
        params[key] = val
    return params
