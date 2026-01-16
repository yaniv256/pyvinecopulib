"""Human-readable JSON format for vine copulas.

This module provides bidirectional translation between the native vinecopulib
JSON format and a human-readable format suitable for review, editing, and
version control.

The human format uses:
- Full descriptive key names instead of abbreviations
- Named parameters per copula family (theta, rho, delta, nu)
- Edge labels showing variable relationships (e.g., "V1-V2|V3")
- Grouped metadata for cleaner structure
"""

from __future__ import annotations

import json
from typing import Any

# Parameter names by family (order matters - matches native parameter order)
FAMILY_PARAMETERS: dict[str, list[str]] = {
    "Independence": [],
    "Gaussian": ["rho"],
    "Student": ["rho", "nu"],
    "Clayton": ["theta"],
    "Gumbel": ["theta"],
    "Frank": ["theta"],
    "Joe": ["theta"],
    "BB1": ["theta", "delta"],
    "BB6": ["theta", "delta"],
    "BB7": ["theta", "delta"],
    "BB8": ["theta", "delta"],
    "Tawn": ["theta", "psi1", "psi2"],
    "Tawn2": ["theta", "psi1", "psi2"],  # Tawn type 2
    "Tll": [],  # Nonparametric - no named parameters
}

# Abbreviated keys in native format → full names
KEY_MAPPING = {
    "fam": "family",
    "rot": "rotation",
    "par": "parameters",
    "ll": "log_likelihood",
    "nobs": "n_observations",
    "npars": "n_parameters",
    "vt": "variable_types",
}

# Reverse mapping for human → native conversion
KEY_MAPPING_REVERSE = {v: k for k, v in KEY_MAPPING.items()}

# Variable type abbreviations
VAR_TYPE_FULL = {"c": "continuous", "d": "discrete"}
VAR_TYPE_ABBREV = {v: k for k, v in VAR_TYPE_FULL.items()}


def _params_to_named(family: str, par_data: list[float]) -> dict[str, float]:
    """Convert parameter list to named dictionary."""
    param_names = FAMILY_PARAMETERS.get(family, [])
    if not param_names:
        # Unknown family or no parameters - return as indexed
        if not par_data:
            return {}
        return {f"param_{i+1}": v for i, v in enumerate(par_data)}
    return {name: par_data[i] for i, name in enumerate(param_names) if i < len(par_data)}


def _params_from_named(family: str, params: dict[str, float]) -> list[float]:
    """Convert named parameter dictionary back to ordered list."""
    param_names = FAMILY_PARAMETERS.get(family, [])
    if not param_names:
        # Unknown family - try to preserve order from param_1, param_2, etc.
        if not params:
            return []
        indexed = [(k, v) for k, v in params.items() if k.startswith("param_")]
        indexed.sort(key=lambda x: int(x[0].split("_")[1]))
        return [v for _, v in indexed]
    return [params[name] for name in param_names if name in params]


def _get_edge_label(
    tree: int,
    edge: int,
    structure: dict[str, Any],
    var_names: list[str] | None = None,
) -> str:
    """Generate human-readable edge label like 'V1-V2' or 'V1-V2|V3'.

    For now, returns generic labels. Full implementation would decode
    the R-vine structure to get actual variable indices.
    """
    # TODO: Decode R-vine structure to get actual conditioned/conditioning sets
    # For now, use placeholder labels
    if tree == 0:
        return f"edge_{edge + 1}"
    else:
        return f"edge_{edge + 1}|tree_{tree + 1}"


def native_to_human(native_json: str | dict) -> dict[str, Any]:
    """Convert native vinecopulib JSON to human-readable format.

    Parameters
    ----------
    native_json : str or dict
        Native JSON string or already-parsed dictionary from Vinecop.to_json()

    Returns
    -------
    dict
        Human-readable dictionary structure
    """
    if isinstance(native_json, str):
        native = json.loads(native_json)
    else:
        native = native_json

    # Extract structure info
    structure = native.get("structure", {})
    order = structure.get("order", [])
    dim = len(order) if order else structure.get("array", {}).get("d", 0)

    # Build variable names
    var_names = [f"V{i}" for i in order] if order else [f"V{i+1}" for i in range(dim)]

    # Convert variable types
    var_types_abbrev = native.get("var_types", [])
    var_types_full = [VAR_TYPE_FULL.get(vt, vt) for vt in var_types_abbrev]

    # Build human-readable output
    human: dict[str, Any] = {
        "dimension": dim,
        "variable_order": order,
        "variable_types": var_types_full,
    }

    # Fit metadata
    if "nobs_" in native or "loglik" in native:
        human["fit_info"] = {}
        if "nobs_" in native:
            human["fit_info"]["n_observations"] = native["nobs_"]
        if "loglik" in native:
            human["fit_info"]["log_likelihood"] = native["loglik"]
        if "threshold" in native:
            human["fit_info"]["threshold"] = native["threshold"]

    # Convert pair copulas
    pair_copulas = native.get("pair copulas", {})
    trees = []

    # Sort tree keys numerically
    tree_keys = sorted(pair_copulas.keys(), key=lambda x: int(x.replace("tree", "")))

    for tree_idx, tree_key in enumerate(tree_keys):
        tree_data = pair_copulas[tree_key]
        edges = []

        # Sort edge keys numerically
        edge_keys = sorted(tree_data.keys(), key=lambda x: int(x.replace("pc", "")))

        for edge_idx, edge_key in enumerate(edge_keys):
            pc = tree_data[edge_key]

            family = pc.get("fam", "Unknown")

            # Extract parameters from nested structure
            par_raw = pc.get("par", {})
            if isinstance(par_raw, dict):
                par_data = par_raw.get("data", [])
            else:
                par_data = par_raw if isinstance(par_raw, list) else []

            edge_info: dict[str, Any] = {
                "edge": _get_edge_label(tree_idx, edge_idx, structure, var_names),
                "family": family,
                "rotation": pc.get("rot", 0),
                "parameters": _params_to_named(family, par_data),
            }

            # Optional fields
            if "ll" in pc:
                edge_info["log_likelihood"] = pc["ll"]
            if "nobs" in pc:
                edge_info["n_observations"] = pc["nobs"]
            if "npars" in pc:
                edge_info["n_parameters"] = pc["npars"]

            edges.append(edge_info)

        trees.append({
            "tree": tree_idx + 1,
            "edges": edges,
        })

    human["trees"] = trees

    # Preserve structure for round-tripping
    human["_structure"] = structure

    return human


def human_to_native(human_json: str | dict) -> dict[str, Any]:
    """Convert human-readable format back to native vinecopulib JSON.

    Parameters
    ----------
    human_json : str or dict
        Human-readable JSON string or dictionary

    Returns
    -------
    dict
        Native vinecopulib dictionary structure (can be saved with Vinecop.from_json)
    """
    if isinstance(human_json, str):
        human = json.loads(human_json)
    else:
        human = human_json

    # Convert variable types back to abbreviations
    var_types_full = human.get("variable_types", [])
    var_types_abbrev = [VAR_TYPE_ABBREV.get(vt, vt) for vt in var_types_full]

    # Build native output
    native: dict[str, Any] = {
        "var_types": var_types_abbrev,
    }

    # Restore structure (required for round-trip)
    if "_structure" in human:
        native["structure"] = human["_structure"]
    else:
        # Reconstruct minimal structure from dimension and order
        order = human.get("variable_order", [])
        dim = human.get("dimension", len(order))
        native["structure"] = {
            "order": order if order else list(range(1, dim + 1)),
            "array": {"d": dim, "t": dim - 1, "data": []},  # Minimal placeholder
        }

    # Restore fit metadata
    fit_info = human.get("fit_info", {})
    if "n_observations" in fit_info:
        native["nobs_"] = fit_info["n_observations"]
    if "log_likelihood" in fit_info:
        native["loglik"] = fit_info["log_likelihood"]
    if "threshold" in fit_info:
        native["threshold"] = fit_info["threshold"]

    # Convert pair copulas back
    pair_copulas: dict[str, Any] = {}

    for tree_data in human.get("trees", []):
        tree_idx = tree_data["tree"] - 1  # Convert to 0-indexed
        tree_key = f"tree{tree_idx}"
        tree_pcs: dict[str, Any] = {}

        for edge_idx, edge_data in enumerate(tree_data.get("edges", [])):
            edge_key = f"pc{edge_idx}"

            family = edge_data.get("family", "Independence")
            params = edge_data.get("parameters", {})
            par_list = _params_from_named(family, params)

            pc: dict[str, Any] = {
                "fam": family,
                "rot": edge_data.get("rotation", 0),
                "par": {
                    "data": par_list,
                    "shape": [len(par_list), 1] if par_list else [0, 0],
                },
                "vt": ["c", "c"],  # Default to continuous
            }

            # Optional fields
            if "log_likelihood" in edge_data:
                pc["ll"] = edge_data["log_likelihood"]
            if "n_observations" in edge_data:
                pc["nobs"] = edge_data["n_observations"]
            if "n_parameters" in edge_data:
                pc["npars"] = edge_data["n_parameters"]

            tree_pcs[edge_key] = pc

        pair_copulas[tree_key] = tree_pcs

    native["pair copulas"] = pair_copulas

    return native


def to_human_json(native_json: str | dict, indent: int = 2) -> str:
    """Convert native JSON to human-readable JSON string.

    Parameters
    ----------
    native_json : str or dict
        Native vinecopulib JSON
    indent : int
        Indentation for pretty-printing (default 2)

    Returns
    -------
    str
        Human-readable JSON string
    """
    human = native_to_human(native_json)
    return json.dumps(human, indent=indent)


def from_human_json(human_json: str | dict) -> str:
    """Convert human-readable JSON back to native format string.

    Parameters
    ----------
    human_json : str or dict
        Human-readable JSON

    Returns
    -------
    str
        Native vinecopulib JSON string
    """
    native = human_to_native(human_json)
    return json.dumps(native)
