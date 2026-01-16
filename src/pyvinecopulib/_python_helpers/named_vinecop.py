"""Vine copula wrapper with variable name support.

This module provides a NamedVinecop class that wraps the C++ Vinecop class
and adds support for variable names. Variable names are automatically
extracted from pandas DataFrames and included in JSON serialization.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Import the C++ Vinecop class
from ..pyvinecopulib_ext import (
    Vinecop as _Vinecop,
    FitControlsVinecop,
    FitControlsBicop,
    RVineStructure,
)

# Try to import pandas, but don't require it
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore
    HAS_PANDAS = False


def _extract_var_names(data: Any) -> list[str] | None:
    """Extract variable names from data if it's a DataFrame."""
    if HAS_PANDAS and isinstance(data, pd.DataFrame):
        return list(data.columns.astype(str))
    return None


def _to_numpy(data: Any) -> NDArray[np.floating[Any]]:
    """Convert data to numpy array, handling DataFrames."""
    if HAS_PANDAS and isinstance(data, pd.DataFrame):
        return data.values.astype(np.float64)
    return np.asarray(data, dtype=np.float64)


class NamedVinecop:
    """Vine copula with variable name support.

    This class wraps the C++ Vinecop class and adds support for variable names.
    When constructed from a pandas DataFrame, column names are automatically
    captured and stored. Variable names are included in JSON serialization
    and used in plotting and human-readable output.

    Parameters
    ----------
    d : int
        Dimension of the vine copula.
    var_names : list[str], optional
        Names for each variable. If not provided, defaults to ["V1", "V2", ...].

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import pyvinecopulib as pv
    >>>
    >>> # Create data with named columns
    >>> df = pd.DataFrame({
    ...     'Stock_A': np.random.randn(100),
    ...     'Stock_B': np.random.randn(100),
    ...     'Stock_C': np.random.randn(100),
    ... })
    >>> u = pd.DataFrame(pv.to_pseudo_obs(df.values), columns=df.columns)
    >>>
    >>> # Fit a vine copula - names are captured automatically
    >>> cop = pv.NamedVinecop.from_data(u)
    >>> print(cop.var_names)
    ['Stock_A', 'Stock_B', 'Stock_C']
    """

    def __init__(
        self,
        d: int,
        var_names: list[str] | None = None,
    ) -> None:
        """Initialize an empty vine copula of given dimension."""
        self._cop = _Vinecop(d)
        self._var_names = var_names or [f"V{i+1}" for i in range(d)]
        if len(self._var_names) != d:
            raise ValueError(
                f"var_names length ({len(self._var_names)}) must match dimension ({d})"
            )

    @classmethod
    def from_data(
        cls,
        data: Any,
        structure: RVineStructure | None = None,
        matrix: NDArray[np.integer[Any]] | None = None,
        var_types: list[str] | None = None,
        controls: FitControlsVinecop | None = None,
        var_names: list[str] | None = None,
    ) -> NamedVinecop:
        """Create and fit a vine copula from data.

        Parameters
        ----------
        data : array-like or DataFrame
            Input data matrix. If a pandas DataFrame, column names are
            automatically used as variable names.
        structure : RVineStructure, optional
            An RVineStructure. Provide either this or `matrix`, but not both.
        matrix : array-like, optional
            RVine matrix. Provide either this or `structure`, but not both.
        var_types : list[str], optional
            Variable types ('c' for continuous, 'd' for discrete).
        controls : FitControlsVinecop, optional
            Fit controls for the vine copula.
        var_names : list[str], optional
            Names for each variable. If not provided and data is a DataFrame,
            column names are used. Otherwise defaults to ["V1", "V2", ...].

        Returns
        -------
        NamedVinecop
            Fitted vine copula with variable names.
        """
        # Extract names from DataFrame if not provided
        if var_names is None:
            var_names = _extract_var_names(data)

        # Convert to numpy
        data_np = _to_numpy(data)
        d = data_np.shape[1]

        # Default var_names if still None
        if var_names is None:
            var_names = [f"V{i+1}" for i in range(d)]

        # Create instance
        instance = cls.__new__(cls)
        instance._var_names = var_names

        # Build kwargs for the C++ factory
        kwargs: dict[str, Any] = {"data": data_np}
        if structure is not None:
            kwargs["structure"] = structure
        if matrix is not None:
            kwargs["matrix"] = matrix
        if var_types is not None:
            kwargs["var_types"] = var_types
        if controls is not None:
            kwargs["controls"] = controls

        instance._cop = _Vinecop.from_data(**kwargs)
        return instance

    @classmethod
    def from_structure(
        cls,
        structure: RVineStructure | None = None,
        matrix: NDArray[np.integer[Any]] | None = None,
        pair_copulas: list[list[Any]] | None = None,
        var_types: list[str] | None = None,
        var_names: list[str] | None = None,
    ) -> NamedVinecop:
        """Create a vine copula from a structure or matrix.

        Parameters
        ----------
        structure : RVineStructure, optional
            An RVineStructure. Provide either this or `matrix`, but not both.
        matrix : array-like, optional
            RVine matrix. Provide either this or `structure`, but not both.
        pair_copulas : list[list[Bicop]], optional
            Pairwise copulas for each edge in the vine.
        var_types : list[str], optional
            Variable types ('c' for continuous, 'd' for discrete).
        var_names : list[str], optional
            Names for each variable.

        Returns
        -------
        NamedVinecop
            Vine copula with the specified structure.
        """
        # Build kwargs for the C++ factory
        kwargs: dict[str, Any] = {}
        if structure is not None:
            kwargs["structure"] = structure
            d = structure.dim
        if matrix is not None:
            kwargs["matrix"] = matrix
            d = matrix.shape[0]
        if pair_copulas is not None:
            kwargs["pair_copulas"] = pair_copulas
        if var_types is not None:
            kwargs["var_types"] = var_types

        cop = _Vinecop.from_structure(**kwargs)

        # Create instance
        instance = cls.__new__(cls)
        instance._cop = cop
        instance._var_names = var_names or [f"V{i+1}" for i in range(d)]
        return instance

    @classmethod
    def from_vinecop(
        cls,
        cop: _Vinecop,
        var_names: list[str] | None = None,
    ) -> NamedVinecop:
        """Wrap an existing Vinecop object with variable names.

        Parameters
        ----------
        cop : Vinecop
            Existing C++ Vinecop object.
        var_names : list[str], optional
            Names for each variable.

        Returns
        -------
        NamedVinecop
            Wrapped vine copula with variable names.
        """
        instance = cls.__new__(cls)
        instance._cop = cop
        instance._var_names = var_names or [f"V{i+1}" for i in range(cop.dim)]
        return instance

    @classmethod
    def from_json(
        cls,
        json_str: str,
        check: bool = True,
    ) -> NamedVinecop:
        """Create a vine copula from a JSON string.

        Parameters
        ----------
        json_str : str
            JSON string (native or human-readable format).
        check : bool
            Whether to validate the structure.

        Returns
        -------
        NamedVinecop
            Vine copula loaded from JSON.
        """
        data = json.loads(json_str)

        # Check if this is human-readable format
        if "trees" in data and "dimension" in data:
            # Human-readable format - convert to native
            from .human_json import human_to_native
            native = human_to_native(data)
            var_names = data.get("var_names")
            json_str = json.dumps(native)
        else:
            # Native format - extract var_names if present
            var_names = data.get("var_names")

        cop = _Vinecop.from_json(json_str, check)
        return cls.from_vinecop(cop, var_names)

    @classmethod
    def from_file(
        cls,
        filename: str,
        check: bool = True,
    ) -> NamedVinecop:
        """Create a vine copula from a JSON file.

        Parameters
        ----------
        filename : str
            Path to the JSON file.
        check : bool
            Whether to validate the structure.

        Returns
        -------
        NamedVinecop
            Vine copula loaded from file.
        """
        with open(filename, "r") as f:
            return cls.from_json(f.read(), check)

    # Properties that delegate to the underlying Vinecop
    @property
    def var_names(self) -> list[str]:
        """Variable names."""
        return self._var_names

    @var_names.setter
    def var_names(self, names: list[str]) -> None:
        """Set variable names."""
        if len(names) != self.dim:
            raise ValueError(
                f"var_names length ({len(names)}) must match dimension ({self.dim})"
            )
        self._var_names = list(names)

    @property
    def dim(self) -> int:
        """Dimension of the vine copula."""
        return self._cop.dim

    @property
    def var_types(self) -> list[str]:
        """Variable types."""
        return self._cop.var_types

    @var_types.setter
    def var_types(self, types: list[str]) -> None:
        """Set variable types."""
        self._cop.var_types = types

    @property
    def trunc_lvl(self) -> int:
        """Truncation level."""
        return self._cop.trunc_lvl

    @property
    def order(self) -> list[int]:
        """R-vine structure order."""
        return self._cop.order

    @property
    def structure(self) -> RVineStructure:
        """R-vine structure."""
        return self._cop.structure

    @property
    def matrix(self) -> NDArray[np.integer[Any]]:
        """R-vine matrix."""
        return self._cop.matrix

    @property
    def pair_copulas(self) -> list[list[Any]]:
        """All pair copulas."""
        return self._cop.pair_copulas

    @property
    def families(self) -> list[list[Any]]:
        """Families of all pair copulas."""
        return self._cop.families

    @property
    def rotations(self) -> list[list[int]]:
        """Rotations of all pair copulas."""
        return self._cop.rotations

    @property
    def parameters(self) -> list[list[NDArray[np.floating[Any]]]]:
        """Parameters of all pair copulas."""
        return self._cop.parameters

    @property
    def taus(self) -> list[list[float]]:
        """Kendall's tau of all pair copulas."""
        return self._cop.taus

    @property
    def npars(self) -> int:
        """Total number of parameters."""
        return self._cop.npars

    @property
    def nobs(self) -> int:
        """Number of observations (for fitted objects)."""
        return self._cop.nobs

    @property
    def threshold(self) -> float:
        """Threshold (for thresholded copulas)."""
        return self._cop.threshold

    # Methods that delegate to the underlying Vinecop
    def select(
        self,
        data: Any,
        controls: FitControlsVinecop | None = None,
    ) -> None:
        """Select the vine copula structure and fit parameters.

        Parameters
        ----------
        data : array-like or DataFrame
            Input data matrix. If a DataFrame, column names update var_names.
        controls : FitControlsVinecop, optional
            Fit controls.
        """
        # Update var_names from DataFrame if applicable
        names = _extract_var_names(data)
        if names is not None:
            self._var_names = names

        data_np = _to_numpy(data)
        if controls is None:
            self._cop.select(data_np)
        else:
            self._cop.select(data_np, controls)

    def fit(
        self,
        data: Any,
        controls: FitControlsBicop | None = None,
        num_threads: int = 1,
    ) -> None:
        """Fit parameters for the current structure.

        Parameters
        ----------
        data : array-like or DataFrame
            Input data matrix.
        controls : FitControlsBicop, optional
            Fit controls.
        num_threads : int
            Number of threads for parallel computation.
        """
        data_np = _to_numpy(data)
        if controls is None:
            self._cop.fit(data_np, FitControlsBicop(), num_threads)
        else:
            self._cop.fit(data_np, controls, num_threads)

    def pdf(self, u: Any, num_threads: int = 1) -> NDArray[np.floating[Any]]:
        """Evaluate the probability density function."""
        return self._cop.pdf(_to_numpy(u), num_threads)

    def cdf(
        self,
        u: Any,
        N: int = 10000,
        num_threads: int = 1,
        seeds: list[int] | None = None,
    ) -> NDArray[np.floating[Any]]:
        """Evaluate the cumulative distribution function."""
        return self._cop.cdf(_to_numpy(u), N, num_threads, seeds or [])

    def simulate(
        self,
        n: int,
        qrng: bool = False,
        num_threads: int = 1,
        seeds: list[int] | None = None,
    ) -> NDArray[np.floating[Any]]:
        """Simulate from the vine copula."""
        return self._cop.simulate(n, qrng, num_threads, seeds or [])

    def rosenblatt(
        self,
        u: Any,
        num_threads: int = 1,
        randomize_discrete: bool = True,
        seeds: list[int] | None = None,
    ) -> NDArray[np.floating[Any]]:
        """Apply the Rosenblatt transform."""
        return self._cop.rosenblatt(
            _to_numpy(u), num_threads, randomize_discrete, seeds or []
        )

    def inverse_rosenblatt(
        self,
        u: Any,
        num_threads: int = 1,
    ) -> NDArray[np.floating[Any]]:
        """Apply the inverse Rosenblatt transform."""
        return self._cop.inverse_rosenblatt(_to_numpy(u), num_threads)

    def loglik(
        self,
        u: Any | None = None,
        num_threads: int = 1,
    ) -> float:
        """Compute the log-likelihood."""
        if u is None:
            return self._cop.loglik()
        return self._cop.loglik(_to_numpy(u), num_threads)

    def aic(self, u: Any | None = None, num_threads: int = 1) -> float:
        """Compute the AIC."""
        if u is None:
            return self._cop.aic()
        return self._cop.aic(_to_numpy(u), num_threads)

    def bic(self, u: Any | None = None, num_threads: int = 1) -> float:
        """Compute the BIC."""
        if u is None:
            return self._cop.bic()
        return self._cop.bic(_to_numpy(u), num_threads)

    def mbicv(
        self,
        u: Any | None = None,
        psi0: float = 0.9,
        num_threads: int = 1,
    ) -> float:
        """Compute the modified BIC for vines."""
        if u is None:
            return self._cop.mbicv()
        return self._cop.mbicv(_to_numpy(u), psi0, num_threads)

    def get_pair_copula(self, tree: int, edge: int) -> Any:
        """Get a specific pair copula."""
        return self._cop.get_pair_copula(tree, edge)

    def get_family(self, tree: int, edge: int) -> Any:
        """Get the family of a pair copula."""
        return self._cop.get_family(tree, edge)

    def get_rotation(self, tree: int, edge: int) -> int:
        """Get the rotation of a pair copula."""
        return self._cop.get_rotation(tree, edge)

    def get_parameters(self, tree: int, edge: int) -> NDArray[np.floating[Any]]:
        """Get the parameters of a pair copula."""
        return self._cop.get_parameters(tree, edge)

    def get_tau(self, tree: int, edge: int) -> float:
        """Get Kendall's tau of a pair copula."""
        return self._cop.get_tau(tree, edge)

    def truncate(self, trunc_lvl: int) -> None:
        """Truncate the vine copula."""
        self._cop.truncate(trunc_lvl)

    def plot(
        self,
        tree: list[int] | None = None,
        add_edge_labels: bool = True,
        layout: str = "graphviz",
    ) -> None:
        """Plot the vine copula structure.

        Uses variable names for node labels automatically.
        """
        self._cop.plot(tree, add_edge_labels, layout, self._var_names)

    def to_json(self, indent: int = -1) -> str:
        """Convert to JSON string with variable names.

        Parameters
        ----------
        indent : int
            Indentation level (-1 for compact).

        Returns
        -------
        str
            JSON string including variable names.
        """
        native_json = self._cop.to_json(indent=-1)
        data = json.loads(native_json)
        data["var_names"] = self._var_names
        if indent < 0:
            return json.dumps(data)
        return json.dumps(data, indent=indent)

    def to_file(self, filename: str, indent: int = -1) -> None:
        """Save to a JSON file with variable names.

        Parameters
        ----------
        filename : str
            Path to the output file.
        indent : int
            Indentation level (-1 for compact).
        """
        with open(filename, "w") as f:
            f.write(self.to_json(indent=indent))

    def to_human_json(self, indent: int = 2) -> str:
        """Convert to human-readable JSON string.

        The output format combines structure and parameters for easy reading:

        {
          "variables": ["Tech_Stock", "Bank_Stock", "Oil_Index", "Gold_Price"],
          "n_observations": 500,
          "log_likelihood": 128.42,
          "trees": {
            "1": {
              "Tech_Stock-Bank_Stock": "Gaussian(rho=0.444)",
              "Bank_Stock-Oil_Index": "Frank(theta=2.394)",
              ...
            },
            "2": { ... }
          }
        }

        Parameters
        ----------
        indent : int
            Indentation level.

        Returns
        -------
        str
            Human-readable JSON string with variable names in edge labels.
        """
        from .human_json import native_to_human, _format_human_json
        native = json.loads(self._cop.to_json())
        human = native_to_human(native, var_names=self._var_names)
        return _format_human_json(human, indent=indent)

    def format(self, trees: list[int] | None = None) -> str:
        """Format the vine copula as a string."""
        return self._cop.format(trees or [])

    def __repr__(self) -> str:
        names_str = ", ".join(self._var_names[:3])
        if len(self._var_names) > 3:
            names_str += ", ..."
        return f"<pyvinecopulib.NamedVinecop(dim={self.dim}, vars=[{names_str}])>"

    def __str__(self) -> str:
        return self.__repr__()

    # Allow access to the underlying Vinecop
    @property
    def _vinecop(self) -> _Vinecop:
        """Access the underlying C++ Vinecop object."""
        return self._cop
