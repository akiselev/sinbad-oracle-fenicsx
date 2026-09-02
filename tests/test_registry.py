"""Offline tests for the capability registry (SV0-C5 D1-D5 coverage table)."""

from pathlib import Path

import pytest

from sinbad_oracle_fenicsx import registry
from sinbad_oracle_fenicsx.normalization_doc import render_normalization_doc

REPO = Path(__file__).resolve().parents[1]
SINBAD_CASES = REPO.parent / "sinbad" / "cases"

EXPECTED = {
    "poisson": ("01-poisson", "Poisson", 2),
    "nonlinear_heat": ("03-nonlinear-heat", "NonlinearHeat", 2),
    "linear_elasticity": ("17-linear-elasticity", "LinearElasticity", 3),
    "stokes": ("25-stokes", "StokesFlow", 2),
    "mixed_darcy": ("13-mixed-darcy", "MixedDarcy", 3),
}


def test_registry_covers_exactly_d1_to_d5():
    assert set(registry.CAPABILITIES) == set(EXPECTED)
    for capability, (case, model, dimension) in EXPECTED.items():
        spec = registry.CAPABILITIES[capability]
        assert spec.capability == capability
        assert spec.module == capability
        assert (spec.sinbad_case, spec.model, spec.dimension) == (case, model, dimension)
        assert spec.observables, capability
        assert all(definition.strip() for definition in spec.normalization.values())


def test_every_capability_module_exists_as_a_file():
    package = REPO / "src" / "sinbad_oracle_fenicsx"
    for spec in registry.CAPABILITIES.values():
        assert (package / f"{spec.module}.py").is_file()


def test_lookup_is_exact_and_returns_none_for_unknown_ids():
    assert registry.lookup("poisson") is registry.CAPABILITIES["poisson"]
    assert registry.lookup("Poisson") is None
    assert registry.lookup("elasticity") is None


def test_normalization_doc_is_generated_from_the_registry():
    # NORMALIZATION.md is the human-readable normalization contract (version 1); it must be
    # byte-identical to what the registry renders, so the two can never drift apart.
    rendered = render_normalization_doc()
    assert (REPO / "NORMALIZATION.md").read_text(encoding="utf-8") == rendered


def test_registry_mirrors_the_sinbad_case_files_when_the_sibling_checkout_is_present():
    if not SINBAD_CASES.is_dir():
        pytest.skip("sibling sinbad checkout not present; case-file mirror check skipped")
    for spec in registry.CAPABILITIES.values():
        case_file = SINBAD_CASES / f"{spec.sinbad_case}.toml"
        assert case_file.is_file(), case_file
        text = case_file.read_text(encoding="utf-8")
        assert f'model = "{spec.model}"' in text
        assert f"dimension = {spec.dimension}" in text
