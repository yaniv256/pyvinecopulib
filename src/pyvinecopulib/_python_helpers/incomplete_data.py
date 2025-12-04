"""
Vine copula fitting for incomplete (missing) data with adaptive truncation.

This module provides functionality to fit vine copulas when different variable
pairs have different numbers of observations. It uses all available pairwise
data for Tree 1, all available triples for Tree 2, etc., and truncates
automatically when observations drop below a threshold.
"""

from __future__ import annotations

from typing import Optional

import pyvinecopulib as pv
import numpy as np
from numpy.typing import NDArray


def fit_vine_incomplete(
  data: NDArray[np.floating],
  min_obs: int = 100,
  structure: Optional[NDArray[np.uint64]] = None,
  family_set: Optional[list[pv.BicopFamily]] = None,
  trunc_lvl: Optional[int] = None,
) -> "pv.Vinecop":
  """
  Fit a vine copula to incomplete data with adaptive truncation.

  This function fits vine copulas when different variable pairs have different
  numbers of observations. It maximizes data usage by:
  - Using all available pairwise observations for Tree 1
  - Using all available triple observations for Tree 2
  - Using all available k-tuple observations for Tree k-1
  - Truncating (setting to independence) when observations drop below threshold

  Parameters
  ----------
  data : ndarray of shape (n, d)
      Data matrix on copula scale [0, 1]. Use np.nan for missing values.

  min_obs : int, default=100
      Minimum number of complete observations required to fit a pair-copula.
      Edges with fewer observations are set to independence.

  structure : ndarray of shape (d, d), optional
      R-vine structure matrix. If None, structure is selected using complete
      cases only.

  family_set : list of BicopFamily, optional
      Set of copula families to consider. If None, uses all families.

  trunc_lvl : int, optional
      Maximum truncation level. If None, fits all trees (subject to min_obs).

  Returns
  -------
  vine : Vinecop
      Fitted vine copula model.

  Notes
  -----
  The function proceeds tree by tree:
  1. For each edge, count complete observations for required variables
  2. If count >= min_obs, fit the pair-copula and compute pseudo-observations
  3. If count < min_obs, set to independence copula
  4. Stop early if all edges in a tree are independence

  Examples
  --------
  >>> import numpy as np
  >>> import pyvinecopulib as pv
  >>> from pyvinecopulib._python_helpers.incomplete_data import fit_vine_incomplete
  >>>
  >>> # Create data with missing values
  >>> np.random.seed(42)
  >>> data = np.random.uniform(0, 1, (1000, 5))
  >>> data[500:, 3] = np.nan  # Variable 3 missing for half the data
  >>> data[700:, 4] = np.nan  # Variable 4 missing for 30% of data
  >>>
  >>> # Fit vine with adaptive truncation
  >>> vine = fit_vine_incomplete(data, min_obs=100)
  """
  n, d = data.shape

  if trunc_lvl is None:
    trunc_lvl = d - 1

  # Set up fit controls for pair-copulas
  if family_set is not None:
    bicop_controls = pv.FitControlsBicop(family_set=family_set)
    vinecop_controls = pv.FitControlsVinecop(family_set=family_set)
  else:
    bicop_controls = pv.FitControlsBicop()
    vinecop_controls = pv.FitControlsVinecop()

  # Step 1: Select structure on complete cases if not provided
  if structure is None:
    complete_mask = ~np.any(np.isnan(data), axis=1)
    n_complete = complete_mask.sum()
    if n_complete < min_obs:
      raise ValueError(
        f"Only {n_complete} complete cases available, need at least {min_obs} "
        "for structure selection. Provide a structure matrix or lower min_obs."
      )
    complete_data = data[complete_mask]
    vine_init = pv.Vinecop.from_data(complete_data, controls=vinecop_controls)
    structure = vine_init.matrix
  else:
    structure = np.asarray(structure, dtype=np.uint64)

  # Step 2: Fit tree by tree
  pair_copulas = []

  # Track pseudo-observations for each variable
  # pseudo_obs[i] contains the current h-function transformed values for variable i
  pseudo_obs = data.copy()

  # Track which observations have valid pseudo-obs for each variable
  valid_mask = ~np.isnan(data)

  for tree in range(min(trunc_lvl, d - 1)):
    tree_pcs = []
    n_edges = d - 1 - tree

    # Store updated pseudo-obs for next tree
    next_pseudo_obs = pseudo_obs.copy()
    next_valid_mask = valid_mask.copy()

    all_independence = True

    for edge in range(n_edges):
      # Parse which variables are involved in this edge
      var1, var2, conditioning_set = _parse_edge(structure, tree, edge, d)

      # Find observations complete for all required variables
      vars_needed = [var1, var2] + conditioning_set
      edge_valid = np.all(valid_mask[:, vars_needed], axis=1)
      n_valid = edge_valid.sum()

      if n_valid < min_obs:
        # Not enough observations: set to independence
        bicop = pv.Bicop(family=pv.indep)
      else:
        # Get the pseudo-observations for this pair
        u_pair = np.column_stack(
          [pseudo_obs[edge_valid, var1], pseudo_obs[edge_valid, var2]]
        )

        # Fit pair-copula
        try:
          bicop = pv.Bicop()
          bicop.select(data=u_pair, controls=bicop_controls)
        except Exception:
          # Fallback to independence if fitting fails
          bicop = pv.Bicop(family=pv.indep)

        if bicop.family != pv.indep:
          all_independence = False

        # Compute h-functions for next tree
        # hfunc1: F(u2|u1), hfunc2: F(u1|u2)
        h1 = bicop.hfunc1(u_pair)  # F(var2 | var1)
        h2 = bicop.hfunc2(u_pair)  # F(var1 | var2)

        # Update pseudo-observations
        # The convention depends on the structure matrix
        # For now, update both variables with their conditional transforms
        next_pseudo_obs[edge_valid, var2] = h1
        next_pseudo_obs[edge_valid, var1] = h2

      tree_pcs.append(bicop)

    pair_copulas.append(tree_pcs)
    pseudo_obs = next_pseudo_obs
    valid_mask = next_valid_mask

    # Early termination if all edges are independence
    if all_independence and tree > 0:
      # Fill remaining trees with independence
      for remaining_tree in range(tree + 1, min(trunc_lvl, d - 1)):
        n_remaining_edges = d - 1 - remaining_tree
        pair_copulas.append(
          [pv.Bicop(family=pv.indep) for _ in range(n_remaining_edges)]
        )
      break

  # Ensure we have all trees (pad with independence if needed)
  while len(pair_copulas) < d - 1:
    remaining_tree = len(pair_copulas)
    n_remaining_edges = d - 1 - remaining_tree
    pair_copulas.append(
      [pv.Bicop(family=pv.indep) for _ in range(n_remaining_edges)]
    )

  return pv.Vinecop.from_structure(matrix=structure, pair_copulas=pair_copulas)


def _parse_edge(
  structure: NDArray[np.uint64],
  tree: int,
  edge: int,
  d: int,
) -> tuple[int, int, list[int]]:
  """
  Parse the structure matrix to get variables involved in an edge.

  Parameters
  ----------
  structure : ndarray of shape (d, d)
      R-vine structure matrix.
  tree : int
      Tree index (0-indexed).
  edge : int
      Edge index within the tree (0-indexed).
  d : int
      Dimension of the vine.

  Returns
  -------
  var1 : int
      First conditioned variable (0-indexed).
  var2 : int
      Second conditioned variable (0-indexed).
  conditioning_set : list of int
      Conditioning variables (0-indexed).
  """
  # In the R-vine matrix (pyvinecopulib convention):
  # - Column j corresponds to edge j in tree 0, etc.
  # - The diagonal entry M[d-1-j, j] is the "base" variable
  # - For tree t, edge e:
  #   - First conditioned: M[d-1-e, e] (anti-diagonal)
  #   - Second conditioned: M[t, e]
  #   - Conditioning set: M[t-1, e], M[t-2, e], ..., M[0, e]

  # First conditioned variable (from anti-diagonal)
  var1 = int(structure[d - 1 - edge, edge]) - 1  # Convert to 0-indexed

  # Second conditioned variable
  var2 = int(structure[tree, edge]) - 1  # Convert to 0-indexed

  # Conditioning set (variables above row 'tree' in column 'edge')
  conditioning_set = []
  for row in range(tree - 1, -1, -1):
    cond_var = int(structure[row, edge]) - 1  # Convert to 0-indexed
    if cond_var >= 0:  # Skip zeros (padding)
      conditioning_set.append(cond_var)

  return var1, var2, conditioning_set


def get_complete_counts(
  data: NDArray[np.floating],
  structure: NDArray[np.uint64],
) -> dict[tuple[int, int, tuple[int, ...]], int]:
  """
  Count complete observations for each edge in a vine structure.

  Parameters
  ----------
  data : ndarray of shape (n, d)
      Data matrix with np.nan for missing values.
  structure : ndarray of shape (d, d)
      R-vine structure matrix.

  Returns
  -------
  counts : dict
      Dictionary mapping (tree, edge, conditioning_tuple) to observation count.
  """
  n, d = data.shape
  valid_mask = ~np.isnan(data)
  counts = {}

  for tree in range(d - 1):
    for edge in range(d - 1 - tree):
      var1, var2, conditioning_set = _parse_edge(structure, tree, edge, d)
      vars_needed = [var1, var2] + conditioning_set
      n_valid = np.all(valid_mask[:, vars_needed], axis=1).sum()
      counts[(tree, edge, tuple(conditioning_set))] = n_valid

  return counts
