"""Tests for human-readable JSON conversion."""

import json
import numpy as np
import pytest
import pyvinecopulib as pv
from pyvinecopulib._python_helpers.human_json import (
    native_to_human,
    human_to_native,
    to_human_json,
    from_human_json,
    FAMILY_PARAMETERS,
    _params_to_named,
    _params_from_named,
    _format_copula,
    _parse_copula,
)


class TestParameterNaming:
    """Test parameter name conversion for each family."""

    def test_gaussian_params(self):
        named = _params_to_named("Gaussian", [0.5])
        assert named == {"rho": 0.5}
        assert _params_from_named("Gaussian", named) == [0.5]

    def test_student_params(self):
        named = _params_to_named("Student", [0.75, 4.5])
        assert named == {"rho": 0.75, "nu": 4.5}
        assert _params_from_named("Student", named) == [0.75, 4.5]

    def test_clayton_params(self):
        named = _params_to_named("Clayton", [2.0])
        assert named == {"theta": 2.0}
        assert _params_from_named("Clayton", named) == [2.0]

    def test_gumbel_params(self):
        named = _params_to_named("Gumbel", [1.5])
        assert named == {"theta": 1.5}
        assert _params_from_named("Gumbel", named) == [1.5]

    def test_frank_params(self):
        named = _params_to_named("Frank", [3.0])
        assert named == {"theta": 3.0}
        assert _params_from_named("Frank", named) == [3.0]

    def test_joe_params(self):
        named = _params_to_named("Joe", [1.2])
        assert named == {"theta": 1.2}
        assert _params_from_named("Joe", named) == [1.2]

    def test_bb1_params(self):
        named = _params_to_named("BB1", [0.5, 1.2])
        assert named == {"theta": 0.5, "delta": 1.2}
        assert _params_from_named("BB1", named) == [0.5, 1.2]

    def test_independence_params(self):
        named = _params_to_named("Independence", [])
        assert named == {}
        assert _params_from_named("Independence", named) == []

    def test_unknown_family_params(self):
        """Unknown families should use indexed parameter names."""
        named = _params_to_named("UnknownFamily", [1.0, 2.0, 3.0])
        assert named == {"param_1": 1.0, "param_2": 2.0, "param_3": 3.0}
        assert _params_from_named("UnknownFamily", named) == [1.0, 2.0, 3.0]


class TestCopulaFormatParse:
    """Test copula specification formatting and parsing."""

    def test_format_gaussian(self):
        spec = _format_copula("Gaussian", 0, {"rho": 0.5})
        assert spec == "Gaussian(rho=0.5)"

    def test_format_clayton_with_rotation(self):
        spec = _format_copula("Clayton", 270, {"theta": 1.5})
        assert spec == "Clayton 270°(theta=1.5)"

    def test_format_independence(self):
        spec = _format_copula("Independence", 0, {})
        assert spec == "Independence"

    def test_format_student_two_params(self):
        spec = _format_copula("Student", 0, {"rho": 0.75, "nu": 4.5})
        assert spec == "Student(rho=0.75, nu=4.5)"

    def test_parse_gaussian(self):
        family, rotation, params = _parse_copula("Gaussian(rho=0.5)")
        assert family == "Gaussian"
        assert rotation == 0
        assert params == {"rho": 0.5}

    def test_parse_clayton_with_rotation(self):
        family, rotation, params = _parse_copula("Clayton 270°(theta=1.5)")
        assert family == "Clayton"
        assert rotation == 270
        assert params == {"theta": 1.5}

    def test_parse_independence(self):
        family, rotation, params = _parse_copula("Independence")
        assert family == "Independence"
        assert rotation == 0
        assert params == {}

    def test_parse_student_two_params(self):
        family, rotation, params = _parse_copula("Student(rho=0.75, nu=4.5)")
        assert family == "Student"
        assert rotation == 0
        assert params == {"rho": 0.75, "nu": 4.5}

    def test_roundtrip_format_parse(self):
        """Test that format -> parse gives back original values."""
        test_cases = [
            ("Gaussian", 0, {"rho": 0.444}),
            ("Clayton", 270, {"theta": 1.5}),
            ("Student", 0, {"rho": 0.75, "nu": 4.5}),
            ("Independence", 0, {}),
            ("Gumbel", 90, {"theta": 2.3}),
        ]
        for family, rotation, params in test_cases:
            spec = _format_copula(family, rotation, params)
            f2, r2, p2 = _parse_copula(spec)
            assert f2 == family
            assert r2 == rotation
            for k, v in params.items():
                assert abs(p2[k] - v) < 1e-10


class TestNativeToHuman:
    """Test conversion from native to human-readable format."""

    @pytest.fixture
    def sample_native(self):
        return {
            "loglik": 7.598,
            "nobs_": 500,
            "pair copulas": {
                "tree0": {
                    "pc0": {
                        "fam": "Clayton",
                        "ll": 1.607,
                        "nobs": 500,
                        "npars": 1.0,
                        "par": {"data": [0.0896], "shape": [1, 1]},
                        "rot": 0,
                        "vt": ["c", "c"],
                    },
                    "pc1": {
                        "fam": "Gaussian",
                        "ll": 0.766,
                        "nobs": 500,
                        "npars": 1.0,
                        "par": {"data": [-0.0547], "shape": [1, 1]},
                        "rot": 90,
                        "vt": ["c", "c"],
                    },
                },
                "tree1": {
                    "pc0": {
                        "fam": "Student",
                        "ll": 0.562,
                        "nobs": 500,
                        "npars": 2.0,
                        "par": {"data": [0.75, 4.5], "shape": [2, 1]},
                        "rot": 0,
                        "vt": ["c", "c"],
                    }
                },
            },
            "structure": {
                "order": [3, 1, 2],
                "array": {"d": 3, "t": 2, "data": [[2], []]},
            },
            "threshold": 0.0,
            "var_types": ["c", "c", "c"],
        }

    def test_variables_list(self, sample_native):
        human = native_to_human(sample_native)
        assert "variables" in human
        # Default names when not provided
        assert human["variables"] == ["V3", "V1", "V2"]

    def test_variables_with_custom_names(self, sample_native):
        human = native_to_human(sample_native, var_names=["A", "B", "C"])
        assert human["variables"] == ["A", "B", "C"]

    def test_fit_info_at_top_level(self, sample_native):
        human = native_to_human(sample_native)
        assert human["n_observations"] == 500
        assert human["log_likelihood"] == 7.598

    def test_trees_as_dict(self, sample_native):
        human = native_to_human(sample_native)
        assert isinstance(human["trees"], dict)
        assert "1" in human["trees"]
        assert "2" in human["trees"]

    def test_edges_as_dict_with_copula_specs(self, sample_native):
        human = native_to_human(sample_native)
        tree1 = human["trees"]["1"]
        assert isinstance(tree1, dict)
        # Values should be copula specification strings
        for edge_label, copula_spec in tree1.items():
            assert isinstance(copula_spec, str)
            # Should be parseable
            family, rot, params = _parse_copula(copula_spec)
            assert family in ["Clayton", "Gaussian", "Student", "Independence"]

    def test_matrix_included_for_roundtrip(self, sample_native):
        human = native_to_human(sample_native)
        # New format uses matrix instead of _native blob
        assert "matrix" in human
        # Matrix should be 2D list of integers
        assert isinstance(human["matrix"], list)
        assert all(isinstance(row, list) for row in human["matrix"])

    def test_accepts_json_string(self, sample_native):
        json_str = json.dumps(sample_native)
        human = native_to_human(json_str)
        assert "variables" in human


class TestHumanToNative:
    """Test conversion from human-readable back to native format."""

    @pytest.fixture
    def sample_human_new_format(self):
        """New compact format with matrix."""
        return {
            "variables": ["A", "B", "C"],
            "n_observations": 500,
            "log_likelihood": 7.598,
            # R-vine matrix: anti-diagonal (bottom-left to top-right) gives order [1, 2, 3]
            "matrix": [[1, 1, 3], [2, 2, 0], [1, 0, 0]],
            "trees": {
                "1": {
                    "A-B": "Clayton(theta=0.0896)",
                    "B-C": "Gaussian 90°(rho=-0.0547)",
                },
                "2": {
                    "A-C|B": "Student(rho=0.75, nu=4.5)",
                },
            },
        }

    def test_new_format_restores_structure(self, sample_human_new_format):
        native = human_to_native(sample_human_new_format)
        # Structure should be reconstructed from matrix
        assert native["structure"]["order"] == [1, 2, 3]  # From matrix anti-diagonal
        assert native["structure"]["array"]["d"] == 3

    def test_new_format_restores_var_types(self, sample_human_new_format):
        native = human_to_native(sample_human_new_format)
        assert native["var_types"] == ["c", "c", "c"]

    def test_new_format_restores_fit_metadata(self, sample_human_new_format):
        native = human_to_native(sample_human_new_format)
        assert native["nobs_"] == 500
        assert native["loglik"] == 7.598

    def test_new_format_parses_copula_specs(self, sample_human_new_format):
        native = human_to_native(sample_human_new_format)
        # Clayton
        pc0 = native["pair copulas"]["tree0"]["pc0"]
        assert pc0["fam"] == "Clayton"
        assert pc0["rot"] == 0
        assert abs(pc0["par"]["data"][0] - 0.0896) < 1e-10

        # Gaussian with rotation
        pc1 = native["pair copulas"]["tree0"]["pc1"]
        assert pc1["fam"] == "Gaussian"
        assert pc1["rot"] == 90
        assert abs(pc1["par"]["data"][0] - (-0.0547)) < 1e-10

        # Student
        pc_t2 = native["pair copulas"]["tree1"]["pc0"]
        assert pc_t2["fam"] == "Student"
        assert pc_t2["par"]["data"] == [0.75, 4.5]

    def test_accepts_json_string(self, sample_human_new_format):
        json_str = json.dumps(sample_human_new_format)
        native = human_to_native(json_str)
        assert native["nobs_"] == 500


class TestRoundTrip:
    """Test that native -> human -> native preserves data."""

    @pytest.fixture
    def sample_native(self):
        return {
            "loglik": 7.598,
            "nobs_": 500,
            "pair copulas": {
                "tree0": {
                    "pc0": {
                        "fam": "Clayton",
                        "ll": 1.607,
                        "nobs": 500,
                        "npars": 1.0,
                        "par": {"data": [0.0896], "shape": [1, 1]},
                        "rot": 0,
                        "vt": ["c", "c"],
                    },
                    "pc1": {
                        "fam": "Gaussian",
                        "ll": 0.766,
                        "nobs": 500,
                        "npars": 1.0,
                        "par": {"data": [-0.0547], "shape": [1, 1]},
                        "rot": 180,
                        "vt": ["c", "c"],
                    },
                },
                "tree1": {
                    "pc0": {
                        "fam": "Student",
                        "ll": 0.562,
                        "nobs": 500,
                        "npars": 2.0,
                        "par": {"data": [0.75, 4.5], "shape": [2, 1]},
                        "rot": 0,
                        "vt": ["c", "c"],
                    }
                },
            },
            "structure": {
                "order": [3, 1, 2],
                "array": {"d": 3, "t": 2, "data": [[2], []]},
            },
            "threshold": 0.0,
            "var_types": ["c", "c", "c"],
        }

    def test_roundtrip_preserves_structure(self, sample_native):
        human = native_to_human(sample_native)
        native_back = human_to_native(human)
        # The order must match exactly
        assert native_back["structure"]["order"] == sample_native["structure"]["order"]
        # The array dimensions must match
        assert native_back["structure"]["array"]["d"] == sample_native["structure"]["array"]["d"]
        # The triangular data must match semantically (ignoring trailing zeros)
        orig_data = sample_native["structure"]["array"]["data"]
        back_data = native_back["structure"]["array"]["data"]
        # Compare non-zero elements in each row
        for orig_row, back_row in zip(orig_data, back_data):
            orig_nonzero = [v for v in orig_row if v != 0]
            back_nonzero = [v for v in back_row if v != 0]
            assert orig_nonzero == back_nonzero

    def test_roundtrip_preserves_var_types(self, sample_native):
        human = native_to_human(sample_native)
        native_back = human_to_native(human)
        assert native_back["var_types"] == sample_native["var_types"]

    def test_roundtrip_preserves_fit_metadata(self, sample_native):
        human = native_to_human(sample_native)
        native_back = human_to_native(human)
        assert native_back["nobs_"] == sample_native["nobs_"]
        # Log-likelihood rounded to 4 decimal places in human format
        assert abs(native_back["loglik"] - sample_native["loglik"]) < 0.001

    def test_roundtrip_preserves_families(self, sample_native):
        human = native_to_human(sample_native)
        native_back = human_to_native(human)
        assert (
            native_back["pair copulas"]["tree0"]["pc0"]["fam"]
            == sample_native["pair copulas"]["tree0"]["pc0"]["fam"]
        )
        assert (
            native_back["pair copulas"]["tree0"]["pc1"]["fam"]
            == sample_native["pair copulas"]["tree0"]["pc1"]["fam"]
        )
        assert (
            native_back["pair copulas"]["tree1"]["pc0"]["fam"]
            == sample_native["pair copulas"]["tree1"]["pc0"]["fam"]
        )

    def test_roundtrip_preserves_rotations(self, sample_native):
        human = native_to_human(sample_native)
        native_back = human_to_native(human)
        assert (
            native_back["pair copulas"]["tree0"]["pc1"]["rot"]
            == sample_native["pair copulas"]["tree0"]["pc1"]["rot"]
        )

    def test_roundtrip_preserves_parameters(self, sample_native):
        human = native_to_human(sample_native)
        native_back = human_to_native(human)
        # Parameters may have minor rounding due to 4 sig figs in human format
        orig_par = sample_native["pair copulas"]["tree0"]["pc0"]["par"]["data"]
        back_par = native_back["pair copulas"]["tree0"]["pc0"]["par"]["data"]
        assert np.allclose(orig_par, back_par, rtol=1e-3)


class TestRoundTripWithRealVinecop:
    """Test round-trip with actual fitted Vinecop objects."""

    def test_roundtrip_fitted_vinecop(self):
        """Fit a vine, convert to human, back to native, and verify it loads."""
        np.random.seed(42)
        n, d = 200, 4

        # Create and fit a vine
        structure = pv.RVineStructure.simulate(d)
        true_vine = pv.Vinecop.from_structure(structure)
        u = true_vine.simulate(n, seeds=[42])

        controls = pv.FitControlsVinecop(
            family_set=[pv.gaussian, pv.clayton, pv.gumbel, pv.frank],
            selection_criterion="bic",
        )
        fitted = pv.Vinecop.from_data(u, controls=controls)

        # Get native JSON
        native_json = fitted.to_json()

        # Convert to human and back
        human = native_to_human(native_json)
        native_back_str = from_human_json(human)

        # Load from the round-tripped JSON
        restored = pv.Vinecop.from_json(native_back_str)

        # Verify key properties match
        assert restored.dim == fitted.dim
        assert np.array_equal(restored.order, fitted.order)
        assert restored.trunc_lvl == fitted.trunc_lvl

        # Verify pair copula families match
        for tree in range(fitted.trunc_lvl):
            for edge in range(fitted.dim - tree - 1):
                orig_pc = fitted.get_pair_copula(tree, edge)
                rest_pc = restored.get_pair_copula(tree, edge)
                assert orig_pc.family == rest_pc.family
                assert orig_pc.rotation == rest_pc.rotation
                # Parameters close (may have 4 sig fig rounding)
                assert np.allclose(orig_pc.parameters, rest_pc.parameters, rtol=1e-3)

    def test_roundtrip_preserves_loglik(self):
        """Verify log-likelihood is close after round-trip."""
        np.random.seed(123)
        n, d = 300, 3

        structure = pv.DVineStructure(order=list(range(1, d + 1)))
        true_vine = pv.Vinecop.from_structure(structure)
        u = true_vine.simulate(n, seeds=[123])

        controls = pv.FitControlsVinecop(
            family_set=[pv.gaussian, pv.student, pv.clayton],
            selection_criterion="aic",
        )
        fitted = pv.Vinecop.from_data(u, controls=controls)

        # Round-trip
        native_json = fitted.to_json()
        human = native_to_human(native_json)
        native_back_str = from_human_json(human)
        restored = pv.Vinecop.from_json(native_back_str)

        # Log-likelihoods should be close (small diff due to param rounding)
        orig_ll = fitted.loglik(u)
        rest_ll = restored.loglik(u)
        assert np.isclose(orig_ll, rest_ll, rtol=1e-3)


class TestStringFunctions:
    """Test the string conversion convenience functions."""

    def test_to_human_json_returns_string(self):
        native = {"loglik": 1.0, "nobs_": 100, "pair copulas": {}, "var_types": [], "structure": {"order": [], "array": {"d": 0}}}
        result = to_human_json(native)
        assert isinstance(result, str)
        # Should be valid JSON
        parsed = json.loads(result)
        assert "variables" in parsed

    def test_to_human_json_respects_indent(self):
        native = {"loglik": 1.0, "nobs_": 100, "pair copulas": {}, "var_types": [], "structure": {"order": [], "array": {"d": 0}}}
        result_2 = to_human_json(native, indent=2)
        result_4 = to_human_json(native, indent=4)
        # More indentation = longer string
        assert len(result_4) > len(result_2)

    def test_from_human_json_returns_string(self):
        human = {
            "variables": ["A", "B"],
            "matrix": [[1, 2], [2, 0]],
            "trees": {},
        }
        result = from_human_json(human)
        assert isinstance(result, str)
        # Should be valid JSON
        parsed = json.loads(result)
        assert "var_types" in parsed
