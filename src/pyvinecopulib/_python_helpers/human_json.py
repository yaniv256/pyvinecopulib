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


def _decode_rvine_matrix(structure: dict[str, Any]) -> list[list[int]]:
    """Reconstruct the natural-order R-vine matrix from the structure dict.

    The native format stores the structure in "natural order" where the
    anti-diagonal is always 1, 2, ..., d. The triangular array stores the
    off-diagonal elements in this natural order representation.

    Returns matrix values as natural-order positions (1-based).
    Use the 'order' array to convert positions to actual variable indices.
    """
    order = structure.get("order", [])
    array_data = structure.get("array", {})
    d = array_data.get("d", len(order))
    tri_data = array_data.get("data", [])

    # Build natural-order matrix (d x d)
    # In natural order, the anti-diagonal is always 1, 2, ..., d
    matrix = [[0] * d for _ in range(d)]

    # Fill diagonal with natural-order positions: 1, 2, ..., d
    # Position i+1 goes to matrix[d-1-i][i] (counter-diagonal)
    for i in range(d):
        matrix[d - 1 - i][i] = i + 1

    # Fill off-diagonal from triangular array (already in natural order)
    for row_idx, row_data in enumerate(tri_data):
        for col_idx, val in enumerate(row_data):
            matrix[row_idx][col_idx] = val

    return matrix


def _encode_rvine_matrix(matrix: list[list[int]]) -> dict[str, Any]:
    """Convert an R-vine matrix to the native structure format.

    The R-vine matrix uses actual variable indices (1-based).
    The native format uses "natural order" where the anti-diagonal
    is always 1, 2, ..., d.

    Parameters
    ----------
    matrix : list[list[int]]
        R-vine matrix with actual variable indices (1-based)

    Returns
    -------
    dict
        Native structure dict with 'order' and 'array' fields
    """
    d = len(matrix)
    if d == 0:
        return {"order": [], "array": {"d": 0, "t": 0, "data": []}}

    # Extract order from anti-diagonal (reading from bottom-left to top-right)
    # matrix[d-1-i][i] = variable at position i in the order
    order = []
    for i in range(d):
        order.append(matrix[d - 1 - i][i])

    # Build inverse map: actual_var -> natural_position (1-based)
    var_to_nat = {var: pos + 1 for pos, var in enumerate(order)}

    # Extract triangular array in natural order
    # Convert actual variables to natural-order positions
    tri_data = []
    for row_idx in range(d - 1):
        row_data = []
        for col_idx in range(d - 1 - row_idx):
            actual_var = matrix[row_idx][col_idx]
            if actual_var == 0:
                row_data.append(0)
            else:
                row_data.append(var_to_nat.get(actual_var, actual_var))
        tri_data.append(row_data)

    return {
        "order": order,
        "array": {
            "d": d,
            "t": d - 1,  # Full truncation level
            "data": tri_data,
        },
    }


def _get_edge_label(
    tree: int,
    edge: int,
    structure: dict[str, Any],
    var_names: list[str] | None = None,
) -> str:
    """Generate human-readable edge label like 'V1-V2' or 'V1-V2|V3'.

    Decodes the R-vine structure to get the actual conditioned and
    conditioning variables for each edge.

    The structure is stored in "natural order" where the anti-diagonal
    is 1, 2, ..., d. Matrix values are natural-order positions that must
    be converted to actual variable indices using the order array:
      actual_variable = order[natural_position - 1]

    The algorithm (from vinecopulib docs):
    For edge e in tree t of a d-dimensional vine:
    - M[d-1-e, e] = first conditioned variable (counter-diagonal)
    - M[t, e] = second conditioned variable
    - M[t-1, e], ..., M[0, e] = conditioning set
    """
    order = structure.get("order", [])
    if not order:
        # Fallback if structure is incomplete
        return f"edge_{edge + 1}" if tree == 0 else f"edge_{edge + 1}|tree_{tree + 1}"

    d = len(order)

    # Reconstruct the natural-order matrix
    matrix = _decode_rvine_matrix(structure)

    # Get natural-order positions from matrix
    nat_cond1 = matrix[d - 1 - edge][edge]  # counter-diagonal
    nat_cond2 = matrix[tree][edge]  # row at tree level

    # Conditioning set: all elements above row=tree in column=edge
    nat_conditioning = []
    for row in range(tree):
        val = matrix[row][edge]
        if val != 0:
            nat_conditioning.append(val)

    def nat_pos_to_name(nat_pos: int) -> str:
        """Convert natural-order position to variable name."""
        if nat_pos <= 0 or nat_pos > d:
            return "?"
        # Convert natural position to actual variable index
        actual_var = order[nat_pos - 1]
        if var_names is not None and actual_var <= len(var_names):
            return var_names[actual_var - 1]
        return f"V{actual_var}"

    cond1 = nat_pos_to_name(nat_cond1)
    cond2 = nat_pos_to_name(nat_cond2)
    cond_set = [nat_pos_to_name(p) for p in nat_conditioning]

    if cond_set:
        return f"{cond1}-{cond2}|{','.join(cond_set)}"
    else:
        return f"{cond1}-{cond2}"


def _structure_to_human(
    structure: dict[str, Any],
    var_names: list[str],
) -> dict[str, Any]:
    """Convert R-vine structure to human-readable format with variable names.

    Parameters
    ----------
    structure : dict
        Native structure dict with 'order' and 'array'
    var_names : list[str]
        Variable names (1-indexed by position in order)

    Returns
    -------
    dict
        Human-readable structure with named variables
    """
    order = structure.get("order", [])
    if not order:
        return {}

    d = len(order)

    # Map variable index to name
    def idx_to_name(idx: int) -> str:
        if idx <= 0 or idx > len(var_names):
            return f"V{idx}"
        return var_names[idx - 1]

    # Convert order to names
    order_named = [idx_to_name(idx) for idx in order]

    # Build the human-readable matrix representation
    # Reconstruct the natural-order matrix first
    matrix = _decode_rvine_matrix(structure)

    # Convert matrix entries from natural-order positions to variable names
    matrix_named: list[list[str]] = []
    for row in matrix:
        row_named = []
        for val in row:
            if val == 0:
                row_named.append("")
            else:
                # val is a natural-order position, convert to actual variable
                actual_var = order[val - 1] if val <= d else val
                row_named.append(idx_to_name(actual_var))
        matrix_named.append(row_named)

    return {
        "variable_order": order_named,
        "matrix": matrix_named,
    }


def _format_copula(
    family: str,
    rotation: int,
    params: dict[str, float],
) -> str:
    """Format copula specification as a compact string.

    Examples:
        "Gaussian(rho=0.444)"
        "Clayton 270°(theta=0.354)"
        "Independence"
    """
    # Build rotation string
    rot_str = f" {rotation}°" if rotation else ""

    # Build parameters string
    if params:
        param_names = FAMILY_PARAMETERS.get(family, [])
        if param_names:
            # Use named parameters in correct order
            param_parts = []
            for name in param_names:
                if name in params:
                    param_parts.append(f"{name}={params[name]:.4g}")
            params_str = ", ".join(param_parts)
        else:
            # Unknown family - show as-is
            params_str = ", ".join(f"{k}={v:.4g}" for k, v in params.items())
        return f"{family}{rot_str}({params_str})"
    else:
        return f"{family}{rot_str}" if rot_str else family


def _parse_copula(spec: str) -> tuple[str, int, dict[str, float]]:
    """Parse copula specification string back to components.

    Examples:
        "Gaussian(rho=0.444)" -> ("Gaussian", 0, {"rho": 0.444})
        "Clayton 270°(theta=0.354)" -> ("Clayton", 270, {"theta": 0.354})
        "Independence" -> ("Independence", 0, {})
    """
    import re

    # Pattern: Family [rotation°][(params)]
    # e.g., "Gaussian(rho=0.444)" or "Clayton 270°(theta=0.354)" or "Independence"
    pattern = r"^(\w+)(?:\s+(\d+)°)?\s*(?:\(([^)]*)\))?$"
    match = re.match(pattern, spec.strip())

    if not match:
        return ("Unknown", 0, {})

    family = match.group(1)
    rotation = int(match.group(2)) if match.group(2) else 0
    params_str = match.group(3)

    params: dict[str, float] = {}
    if params_str:
        # Parse "name=value, name2=value2"
        for part in params_str.split(","):
            part = part.strip()
            if "=" in part:
                name, val = part.split("=", 1)
                params[name.strip()] = float(val.strip())

    return (family, rotation, params)


def native_to_human(
    native_json: str | dict,
    var_names: list[str] | None = None,
) -> dict[str, Any]:
    """Convert native vinecopulib JSON to human-readable format.

    The output format combines structure and parameters for easy reading:

    {
      "variables": ["Tech_Stock", "Bank_Stock", "Oil_Index", "Gold_Price"],
      "matrix": [[1, 1, 4, 4], [2, 4, 1, 0], [4, 2, 0, 0], [3, 0, 0, 0]],
      "trees": {
        "1": {
          "Tech_Stock-Bank_Stock": "Gaussian(rho=0.444)",
          "Bank_Stock-Oil_Index": "Frank(theta=2.394)",
          "Oil_Index-Gold_Price": "Clayton 270°(theta=0.354)"
        },
        "2": {
          "Tech_Stock-Oil_Index|Bank_Stock": "Clayton(theta=0.314)",
          ...
        }
      }
    }

    The ``matrix`` field stores the R-vine structure matrix, which fully
    defines the vine structure. This is more readable than the internal
    representation and sufficient for reconstruction.

    Parameters
    ----------
    native_json : str or dict
        Native JSON string or already-parsed dictionary from Vinecop.to_json()
    var_names : list[str], optional
        Custom variable names. If not provided, uses "V1", "V2", etc.

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

    # Use provided var_names, or from JSON, or generate default
    if var_names is None:
        var_names = native.get("var_names")
    if var_names is None:
        var_names = [f"V{i}" for i in order] if order else [f"V{i+1}" for i in range(dim)]

    # Build human-readable output
    human: dict[str, Any] = {
        "variables": var_names,
    }

    # Fit metadata (compact)
    if "nobs_" in native:
        human["n_observations"] = native["nobs_"]
    if "loglik" in native:
        human["log_likelihood"] = round(native["loglik"], 4)

    # Store the R-vine matrix (reconstructed from structure)
    # This is sufficient to reconstruct the vine structure
    if structure:
        matrix = _decode_rvine_matrix(structure)
        # Convert natural-order positions to actual variable indices
        rvine_matrix = []
        for row in matrix:
            actual_row = []
            for val in row:
                if val == 0:
                    actual_row.append(0)
                else:
                    # val is natural-order position, convert to actual index
                    actual_row.append(order[val - 1] if val <= len(order) else val)
            rvine_matrix.append(actual_row)
        human["matrix"] = rvine_matrix

    # Store var_types if not all continuous
    var_types = native.get("var_types", [])
    if var_types and not all(vt == "c" for vt in var_types):
        human["var_types"] = var_types

    # Convert pair copulas to compact tree structure
    pair_copulas = native.get("pair copulas", {})
    trees: dict[str, dict[str, str]] = {}

    # Sort tree keys numerically
    tree_keys = sorted(pair_copulas.keys(), key=lambda x: int(x.replace("tree", "")))

    for tree_idx, tree_key in enumerate(tree_keys):
        tree_data = pair_copulas[tree_key]
        tree_edges: dict[str, str] = {}

        # Sort edge keys numerically
        edge_keys = sorted(tree_data.keys(), key=lambda x: int(x.replace("pc", "")))

        for edge_idx, edge_key in enumerate(edge_keys):
            pc = tree_data[edge_key]

            family = pc.get("fam", "Unknown")
            rotation = pc.get("rot", 0)

            # Extract parameters from nested structure
            par_raw = pc.get("par", {})
            if isinstance(par_raw, dict):
                par_data = par_raw.get("data", [])
            else:
                par_data = par_raw if isinstance(par_raw, list) else []

            params = _params_to_named(family, par_data)
            edge_label = _get_edge_label(tree_idx, edge_idx, structure, var_names)
            copula_spec = _format_copula(family, rotation, params)

            tree_edges[edge_label] = copula_spec

        trees[str(tree_idx + 1)] = tree_edges

    human["trees"] = trees

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

    # Build native output
    native: dict[str, Any] = {}

    # Determine dimension and var_types
    variables = human.get("variables", [])
    dim = len(variables)

    # Reconstruct structure from matrix
    if "matrix" in human:
        native["structure"] = _encode_rvine_matrix(human["matrix"])
        native["var_types"] = human.get("var_types", ["c"] * dim)
    else:
        raise ValueError("Human JSON must contain 'matrix' field")

    # Restore fit metadata
    if "n_observations" in human:
        native["nobs_"] = human["n_observations"]
    if "log_likelihood" in human:
        native["loglik"] = human["log_likelihood"]

    # Convert pair copulas back
    pair_copulas: dict[str, Any] = {}
    trees = human.get("trees", {})

    for tree_key, tree_edges in trees.items():
        tree_idx = int(tree_key) - 1
        native_tree_key = f"tree{tree_idx}"
        tree_pcs: dict[str, Any] = {}

        for edge_idx, (edge_label, copula_spec) in enumerate(tree_edges.items()):
            edge_key = f"pc{edge_idx}"

            family, rotation, params = _parse_copula(copula_spec)
            par_list = _params_from_named(family, params)

            pc: dict[str, Any] = {
                "fam": family,
                "rot": rotation,
                "par": {
                    "data": par_list,
                    "shape": [len(par_list), 1] if par_list else [0, 0],
                },
                "vt": ["c", "c"],
            }
            tree_pcs[edge_key] = pc

        pair_copulas[native_tree_key] = tree_pcs

    native["pair copulas"] = pair_copulas

    return native


def _format_human_json(human: dict, indent: int = 2) -> str:
    """Format human-readable dict as JSON with compact matrix rows.

    Standard json.dumps expands matrix rows vertically, making them hard to read.
    This function keeps matrix rows on single lines for readability.
    """
    lines = ["{"]
    ind = " " * indent

    keys = list(human.keys())
    for i, key in enumerate(keys):
        value = human[key]
        comma = "," if i < len(keys) - 1 else ""

        if key == "matrix":
            # Format matrix with each row on one line
            lines.append(f'{ind}"{key}": [')
            for j, row in enumerate(value):
                row_comma = "," if j < len(value) - 1 else ""
                row_str = "[" + ", ".join(str(x) for x in row) + "]"
                lines.append(f"{ind}{ind}{row_str}{row_comma}")
            lines.append(f"{ind}]{comma}")
        elif key == "trees":
            # Format trees with nested structure
            lines.append(f'{ind}"{key}": {{')
            tree_keys = list(value.keys())
            for j, tree_key in enumerate(tree_keys):
                tree_comma = "," if j < len(tree_keys) - 1 else ""
                edges = value[tree_key]
                lines.append(f'{ind}{ind}"{tree_key}": {{')
                edge_items = list(edges.items())
                for k, (edge, spec) in enumerate(edge_items):
                    edge_comma = "," if k < len(edge_items) - 1 else ""
                    lines.append(f'{ind}{ind}{ind}"{edge}": "{spec}"{edge_comma}')
                lines.append(f"{ind}{ind}}}{tree_comma}")
            lines.append(f"{ind}}}{comma}")
        elif isinstance(value, list) and all(isinstance(x, str) for x in value):
            # Format string lists (like variables) on one line if short enough
            list_str = json.dumps(value, ensure_ascii=False)
            if len(list_str) < 60:
                lines.append(f'{ind}"{key}": {list_str}{comma}')
            else:
                lines.append(f'{ind}"{key}": {json.dumps(value, indent=indent, ensure_ascii=False)}{comma}')
        else:
            # Default formatting
            val_str = json.dumps(value, ensure_ascii=False)
            lines.append(f'{ind}"{key}": {val_str}{comma}')

    lines.append("}")
    return "\n".join(lines)


def to_human_json(
    native_json: str | dict,
    indent: int = 2,
    var_names: list[str] | None = None,
) -> str:
    """Convert native JSON to human-readable JSON string.

    Parameters
    ----------
    native_json : str or dict
        Native vinecopulib JSON
    indent : int
        Indentation for pretty-printing (default 2)
    var_names : list[str], optional
        Custom variable names for edge labels

    Returns
    -------
    str
        Human-readable JSON string
    """
    human = native_to_human(native_json, var_names=var_names)
    return _format_human_json(human, indent=indent)


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
