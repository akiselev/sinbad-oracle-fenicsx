"""Independent SHOW-2 trajectory and reaction balance, no Sinbad numbers."""

import pytest

pytest.importorskip("dolfinx")
from sinbad_oracle_fenicsx.electrothermal_component import solve  # noqa: E402


def test_heating_reactions_balance_every_accepted_step():
    result = solve((2, 2, 2))
    steps = result.notes["trajectory"]
    assert [p["time"] for p in steps] == [0.25, 0.5, 0.75, 1.0]
    assert all(abs(p["balance_defect"]) < 1e-9 for p in steps)
    assert all(p["cooling_reaction"] > 0 for p in steps)
    assert all(p["electrical_power"] > p["energy_rate"] > 0 for p in steps)
    assert all(a["temperature_max"] < b["temperature_max"] for a, b in zip(steps, steps[1:]))
    assert all(a["electrical_power"] > b["electrical_power"] for a, b in zip(steps, steps[1:]))
    assert 300 < min(result.fields[1].values) + 1e-10
    assert 300 < max(result.fields[1].values) < 325
