"""Independent submesh reference against a hand-derived resistance solution."""

import pytest

pytest.importorskip("dolfinx")
from sinbad_oracle_fenicsx.two_material_conduction import solve  # noqa: E402


def test_separate_submesh_heat_flow_matches_series_resistances():
    for n in [1, 2, 4]:
        result = solve((n, n, n))
        assert abs(result.observables["interface_temperature"] - 320) < 1e-9
        assert abs(result.observables["heat_flow"] - 160) < 1e-9
        assert max(abs(q - 160) for q in result.notes["component_heat_flow"]) < 1e-9
