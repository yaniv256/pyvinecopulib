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
                "order": [3, 1, 2, 4],
                "array": {"d": 4, "t": 3, "data": []},
            },
            "threshold": 0.0,
            "var_types": ["c", "c", "c", "c"],
        }

    def test_dimension_extracted(self, sample_native):
        human = native_to_human(sample_native)
        assert human["dimension"] == 4

    def test_variable_order_preserved(self, sample_native):
        human = native_to_human(sample_native)
        assert human["variable_order"] == [3, 1, 2, 4]

    def test_variable_types_expanded(self, sample_native):
        human = native_to_human(sample_native)
        assert human["variable_types"] == [
            "continuous",
            "continuous",
            "continuous",
            "continuous",
        ]

    def test_fit_info_grouped(self, sample_native):
        human = native_to_human(sample_native)
        assert "fit_info" in human
        assert human["fit_info"]["n_observations"] == 500
        assert human["fit_info"]["log_likelihood"] == 7.598
        assert human["fit_info"]["threshold"] == 0.0

    def test_trees_converted(self, sample_native):
        human = native_to_human(sample_native)
        assert len(human["trees"]) == 2
        assert human["trees"][0]["tree"] == 1
        assert human["trees"][1]["tree"] == 2

    def test_edges_have_named_parameters(self, sample_native):
        human = native_to_human(sample_native)
        # Tree 1, edge 1: Clayton
        edge0 = human["trees"][0]["edges"][0]
        assert edge0["family"] == "Clayton"
        assert edge0["parameters"] == {"theta": 0.0896}

        # Tree 1, edge 2: Gaussian
        edge1 = human["trees"][0]["edges"][1]
        assert edge1["family"] == "Gaussian"
        assert edge1["parameters"] == {"rho": -0.0547}
        assert edge1["rotation"] == 90

        # Tree 2, edge 1: Student
        edge2 = human["trees"][1]["edges"][0]
        assert edge2["family"] == "Student"
        assert edge2["parameters"] == {"rho": 0.75, "nu": 4.5}

    def test_structure_preserved_for_roundtrip(self, sample_native):
        human = native_to_human(sample_native)
        assert "_structure" in human
        assert human["_structure"] == sample_native["structure"]

    def test_accepts_json_string(self, sample_native):
        json_str = json.dumps(sample_native)
        human = native_to_human(json_str)
        assert human["dimension"] == 4


class TestHumanToNative:
    """Test conversion from human-readable back to native format."""

    @pytest.fixture
    def sample_human(self):
        return {
            "dimension": 4,
            "variable_order": [3, 1, 2, 4],
            "variable_types": ["continuous", "continuous", "continuous", "continuous"],
            "fit_info": {
                "n_observations": 500,
                "log_likelihood": 7.598,
                "threshold": 0.0,
            },
            "trees": [
                {
                    "tree": 1,
                    "edges": [
                        {
                            "edge": "edge_1",
                            "family": "Clayton",
                            "rotation": 0,
                            "parameters": {"theta": 0.0896},
                            "log_likelihood": 1.607,
                            "n_observations": 500,
                            "n_parameters": 1.0,
                        },
                        {
                            "edge": "edge_2",
                            "family": "Gaussian",
                            "rotation": 90,
                            "parameters": {"rho": -0.0547},
                            "log_likelihood": 0.766,
                            "n_observations": 500,
                            "n_parameters": 1.0,
                        },
                    ],
                },
                {
                    "tree": 2,
                    "edges": [
                        {
                            "edge": "edge_1|tree_2",
                            "family": "Student",
                            "rotation": 0,
                            "parameters": {"rho": 0.75, "nu": 4.5},
                            "log_likelihood": 0.562,
                            "n_observations": 500,
                            "n_parameters": 2.0,
                        }
                    ],
                },
            ],
            "_structure": {
                "order": [3, 1, 2, 4],
                "array": {"d": 4, "t": 3, "data": []},
            },
        }

    def test_var_types_abbreviated(self, sample_human):
        native = human_to_native(sample_human)
        assert native["var_types"] == ["c", "c", "c", "c"]

    def test_fit_info_ungrouped(self, sample_human):
        native = human_to_native(sample_human)
        assert native["nobs_"] == 500
        assert native["loglik"] == 7.598
        assert native["threshold"] == 0.0

    def test_structure_restored(self, sample_human):
        native = human_to_native(sample_human)
        assert native["structure"] == sample_human["_structure"]

    def test_pair_copulas_converted(self, sample_human):
        native = human_to_native(sample_human)
        assert "pair copulas" in native
        assert "tree0" in native["pair copulas"]
        assert "tree1" in native["pair copulas"]

    def test_parameters_converted_to_list(self, sample_human):
        native = human_to_native(sample_human)
        # Clayton
        pc0 = native["pair copulas"]["tree0"]["pc0"]
        assert pc0["fam"] == "Clayton"
        assert pc0["par"]["data"] == [0.0896]

        # Student
        pc_student = native["pair copulas"]["tree1"]["pc0"]
        assert pc_student["fam"] == "Student"
        assert pc_student["par"]["data"] == [0.75, 4.5]

    def test_rotation_preserved(self, sample_human):
        native = human_to_native(sample_human)
        pc1 = native["pair copulas"]["tree0"]["pc1"]
        assert pc1["rot"] == 90

    def test_accepts_json_string(self, sample_human):
        json_str = json.dumps(sample_human)
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
                "order": [3, 1, 2, 4],
                "array": {"d": 4, "t": 3, "data": [[2, 4, 4], [4, 3], [3]]},
            },
            "threshold": 0.0,
            "var_types": ["c", "c", "c", "c"],
        }

    def test_roundtrip_preserves_structure(self, sample_native):
        human = native_to_human(sample_native)
        native_back = human_to_native(human)
        assert native_back["structure"] == sample_native["structure"]

    def test_roundtrip_preserves_var_types(self, sample_native):
        human = native_to_human(sample_native)
        native_back = human_to_native(human)
        assert native_back["var_types"] == sample_native["var_types"]

    def test_roundtrip_preserves_fit_metadata(self, sample_native):
        human = native_to_human(sample_native)
        native_back = human_to_native(human)
        assert native_back["nobs_"] == sample_native["nobs_"]
        assert native_back["loglik"] == sample_native["loglik"]
        assert native_back["threshold"] == sample_native["threshold"]

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
        # Clayton theta
        assert (
            native_back["pair copulas"]["tree0"]["pc0"]["par"]["data"]
            == sample_native["pair copulas"]["tree0"]["pc0"]["par"]["data"]
        )
        # Student rho and nu
        assert (
            native_back["pair copulas"]["tree1"]["pc0"]["par"]["data"]
            == sample_native["pair copulas"]["tree1"]["pc0"]["par"]["data"]
        )

    def test_roundtrip_preserves_edge_metadata(self, sample_native):
        human = native_to_human(sample_native)
        native_back = human_to_native(human)
        pc0 = native_back["pair copulas"]["tree0"]["pc0"]
        orig_pc0 = sample_native["pair copulas"]["tree0"]["pc0"]
        assert pc0["ll"] == orig_pc0["ll"]
        assert pc0["nobs"] == orig_pc0["nobs"]
        assert pc0["npars"] == orig_pc0["npars"]


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
                assert np.allclose(orig_pc.parameters, rest_pc.parameters)

    def test_roundtrip_preserves_loglik(self):
        """Verify log-likelihood is the same after round-trip."""
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

        # Log-likelihoods should match
        orig_ll = fitted.loglik(u)
        rest_ll = restored.loglik(u)
        assert np.isclose(orig_ll, rest_ll)


class TestStringFunctions:
    """Test the string conversion convenience functions."""

    def test_to_human_json_returns_string(self):
        native = {"loglik": 1.0, "nobs_": 100, "pair copulas": {}, "var_types": []}
        result = to_human_json(native)
        assert isinstance(result, str)
        # Should be valid JSON
        parsed = json.loads(result)
        assert "fit_info" in parsed

    def test_to_human_json_respects_indent(self):
        native = {"loglik": 1.0, "nobs_": 100, "pair copulas": {}, "var_types": []}
        result_2 = to_human_json(native, indent=2)
        result_4 = to_human_json(native, indent=4)
        # More indentation = longer string
        assert len(result_4) > len(result_2)

    def test_from_human_json_returns_string(self):
        human = {
            "dimension": 2,
            "variable_order": [1, 2],
            "variable_types": ["continuous", "continuous"],
            "trees": [],
            "_structure": {"order": [1, 2], "array": {"d": 2, "t": 1, "data": []}},
        }
        result = from_human_json(human)
        assert isinstance(result, str)
        # Should be valid JSON
        parsed = json.loads(result)
        assert "var_types" in parsed
