"""Tests for NamedVinecop class."""

import json
import numpy as np
import pandas as pd
import pytest
import pyvinecopulib as pv


class TestNamedVinecopConstruction:
    """Test NamedVinecop construction methods."""

    def test_init_default_names(self):
        cop = pv.NamedVinecop(5)
        assert cop.var_names == ["V1", "V2", "V3", "V4", "V5"]
        assert cop.dim == 5

    def test_init_custom_names(self):
        names = ["A", "B", "C"]
        cop = pv.NamedVinecop(3, var_names=names)
        assert cop.var_names == names

    def test_init_wrong_length_raises(self):
        with pytest.raises(ValueError, match="must match dimension"):
            pv.NamedVinecop(3, var_names=["A", "B"])

    def test_from_data_numpy_default_names(self):
        np.random.seed(42)
        u = pv.to_pseudo_obs(np.random.randn(100, 3))
        cop = pv.NamedVinecop.from_data(u)
        assert cop.var_names == ["V1", "V2", "V3"]

    def test_from_data_numpy_custom_names(self):
        np.random.seed(42)
        u = pv.to_pseudo_obs(np.random.randn(100, 3))
        cop = pv.NamedVinecop.from_data(u, var_names=["X", "Y", "Z"])
        assert cop.var_names == ["X", "Y", "Z"]

    def test_from_data_dataframe_extracts_names(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "Stock_A": np.random.randn(100),
            "Stock_B": np.random.randn(100),
            "Stock_C": np.random.randn(100),
        })
        u = pd.DataFrame(pv.to_pseudo_obs(df.values), columns=df.columns)
        cop = pv.NamedVinecop.from_data(u)
        assert cop.var_names == ["Stock_A", "Stock_B", "Stock_C"]

    def test_from_data_explicit_names_override_dataframe(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "A": np.random.randn(100),
            "B": np.random.randn(100),
        })
        u = pd.DataFrame(pv.to_pseudo_obs(df.values), columns=df.columns)
        cop = pv.NamedVinecop.from_data(u, var_names=["X", "Y"])
        assert cop.var_names == ["X", "Y"]

    def test_from_vinecop_wraps_existing(self):
        np.random.seed(42)
        u = pv.to_pseudo_obs(np.random.randn(100, 3))
        base_cop = pv.Vinecop(3)
        base_cop.select(u)

        named = pv.NamedVinecop.from_vinecop(base_cop, var_names=["A", "B", "C"])
        assert named.var_names == ["A", "B", "C"]
        assert named.dim == base_cop.dim


class TestNamedVinecopProperties:
    """Test NamedVinecop properties delegate correctly."""

    @pytest.fixture
    def fitted_cop(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "X": np.random.randn(200),
            "Y": np.random.randn(200),
            "Z": np.random.randn(200),
        })
        u = pd.DataFrame(pv.to_pseudo_obs(df.values), columns=df.columns)
        return pv.NamedVinecop.from_data(u)

    def test_dim(self, fitted_cop):
        assert fitted_cop.dim == 3

    def test_var_types(self, fitted_cop):
        assert len(fitted_cop.var_types) == 3
        assert all(vt == "c" for vt in fitted_cop.var_types)

    def test_trunc_lvl(self, fitted_cop):
        assert fitted_cop.trunc_lvl == 2  # d-1 for 3D

    def test_order(self, fitted_cop):
        assert len(fitted_cop.order) == 3

    def test_npars(self, fitted_cop):
        assert fitted_cop.npars >= 0

    def test_nobs(self, fitted_cop):
        assert fitted_cop.nobs == 200

    def test_var_names_setter(self, fitted_cop):
        fitted_cop.var_names = ["A", "B", "C"]
        assert fitted_cop.var_names == ["A", "B", "C"]

    def test_var_names_setter_wrong_length(self, fitted_cop):
        with pytest.raises(ValueError, match="must match dimension"):
            fitted_cop.var_names = ["A", "B"]


class TestNamedVinecopSerialization:
    """Test JSON serialization with variable names."""

    @pytest.fixture
    def fitted_cop(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "Temp": np.random.randn(200),
            "Pressure": np.random.randn(200),
            "Humidity": np.random.randn(200),
        })
        u = pd.DataFrame(pv.to_pseudo_obs(df.values), columns=df.columns)
        return pv.NamedVinecop.from_data(u)

    def test_to_json_includes_var_names(self, fitted_cop):
        json_str = fitted_cop.to_json()
        data = json.loads(json_str)
        assert "var_names" in data
        assert data["var_names"] == ["Temp", "Pressure", "Humidity"]

    def test_to_json_indent(self, fitted_cop):
        json_str = fitted_cop.to_json(indent=2)
        assert "\n" in json_str

    def test_from_json_restores_var_names(self, fitted_cop):
        json_str = fitted_cop.to_json()
        loaded = pv.NamedVinecop.from_json(json_str)
        assert loaded.var_names == fitted_cop.var_names

    def test_from_json_roundtrip_preserves_dim(self, fitted_cop):
        json_str = fitted_cop.to_json()
        loaded = pv.NamedVinecop.from_json(json_str)
        assert loaded.dim == fitted_cop.dim

    def test_to_human_json_uses_var_names_in_edges(self, fitted_cop):
        human_json = fitted_cop.to_human_json()
        data = json.loads(human_json)
        # New format: trees is dict, edges are keys
        tree1 = data["trees"]["1"]
        first_edge = list(tree1.keys())[0]
        assert any(name in first_edge for name in fitted_cop.var_names)

    def test_to_human_json_includes_variables(self, fitted_cop):
        human_json = fitted_cop.to_human_json()
        data = json.loads(human_json)
        assert "variables" in data
        assert data["variables"] == ["Temp", "Pressure", "Humidity"]

    def test_to_human_json_compact_format(self, fitted_cop):
        """Test the new compact JSON format."""
        human_json = fitted_cop.to_human_json()
        data = json.loads(human_json)
        # Trees should be dict with string keys
        assert isinstance(data["trees"], dict)
        assert "1" in data["trees"]
        # Each tree should be dict of edge -> copula_spec
        tree1 = data["trees"]["1"]
        assert isinstance(tree1, dict)
        for edge, spec in tree1.items():
            assert isinstance(spec, str)
            # Spec should contain family name
            assert any(fam in spec for fam in ["Gaussian", "Clayton", "Gumbel", "Frank", "Student", "Independence"])


class TestNamedVinecopMethods:
    """Test NamedVinecop methods work correctly."""

    def test_select_updates_var_names_from_dataframe(self):
        cop = pv.NamedVinecop(3)
        assert cop.var_names == ["V1", "V2", "V3"]

        np.random.seed(42)
        df = pd.DataFrame({
            "A": np.random.randn(100),
            "B": np.random.randn(100),
            "C": np.random.randn(100),
        })
        u = pd.DataFrame(pv.to_pseudo_obs(df.values), columns=df.columns)
        cop.select(u)
        assert cop.var_names == ["A", "B", "C"]

    def test_pdf_works(self):
        np.random.seed(42)
        u = pv.to_pseudo_obs(np.random.randn(100, 3))
        cop = pv.NamedVinecop.from_data(u)
        pdf_vals = cop.pdf(u[:10])
        assert len(pdf_vals) == 10
        assert all(p > 0 for p in pdf_vals)

    def test_simulate_works(self):
        np.random.seed(42)
        u = pv.to_pseudo_obs(np.random.randn(100, 3))
        cop = pv.NamedVinecop.from_data(u)
        sim = cop.simulate(50)
        assert sim.shape == (50, 3)
        assert sim.min() >= 0
        assert sim.max() <= 1

    def test_loglik_works(self):
        np.random.seed(42)
        u = pv.to_pseudo_obs(np.random.randn(100, 3))
        cop = pv.NamedVinecop.from_data(u)
        ll = cop.loglik()
        assert isinstance(ll, float)


class TestNativeToHumanWithVarNames:
    """Test native_to_human function with var_names parameter."""

    def test_uses_provided_var_names(self):
        np.random.seed(42)
        u = pv.to_pseudo_obs(np.random.randn(100, 3))
        cop = pv.Vinecop(3)
        cop.select(u)

        human = pv.native_to_human(
            json.loads(cop.to_json()),
            var_names=["Alpha", "Beta", "Gamma"]
        )
        # New format: trees is dict, edges are keys
        tree1 = human["trees"]["1"]
        first_edge = list(tree1.keys())[0]
        assert any(name in first_edge for name in ["Alpha", "Beta", "Gamma"])

    def test_reads_var_names_from_json(self):
        np.random.seed(42)
        u = pv.to_pseudo_obs(np.random.randn(100, 3))
        cop = pv.Vinecop(3)
        cop.select(u)

        native = json.loads(cop.to_json())
        native["var_names"] = ["X", "Y", "Z"]

        human = pv.native_to_human(native)
        tree1 = human["trees"]["1"]
        first_edge = list(tree1.keys())[0]
        assert any(name in first_edge for name in ["X", "Y", "Z"])

    def test_explicit_var_names_override_json(self):
        np.random.seed(42)
        u = pv.to_pseudo_obs(np.random.randn(100, 3))
        cop = pv.Vinecop(3)
        cop.select(u)

        native = json.loads(cop.to_json())
        native["var_names"] = ["X", "Y", "Z"]

        human = pv.native_to_human(native, var_names=["A", "B", "C"])
        tree1 = human["trees"]["1"]
        first_edge = list(tree1.keys())[0]
        # Explicit var_names should override
        assert any(name in first_edge for name in ["A", "B", "C"])
        assert not any(name in first_edge for name in ["X", "Y", "Z"])
