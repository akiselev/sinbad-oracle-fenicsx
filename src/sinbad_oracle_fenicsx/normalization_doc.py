"""Renders NORMALIZATION.md from the registry so the document cannot drift from the code."""

from __future__ import annotations

from . import registry

NORMALIZATION_VERSION = 1


def render_normalization_doc() -> str:
    lines = [
        "# Normalization contract",
        "",
        "<!-- Generated from src/sinbad_oracle_fenicsx/registry.py by "
        "`python -m sinbad_oracle_fenicsx.normalization_doc`; do not edit by hand. -->",
        "",
        f"`OracleToolIdentity.normalization_version = {NORMALIZATION_VERSION}`.",
        "",
        "Every observable is one finite IEEE-754 double (wire form: the `FiniteF64` bit pattern,",
        "see `protocol.py`) in the SI unit system the mirrored Sinbad case authors, computed by",
        "exact quadrature of the discrete solution on the mesh the adapter itself generated",
        "(Finitum's `simplex_box` layout: unit box, `[nx, ny]` bricks each split along the",
        "lower-left/upper-right diagonal in 2-D, six Kuhn tetrahedra per brick in 3-D).",
        "Observable ids are the keys of `OracleResult.observables`; where an id matches a",
        "Sinbad `[[observables]]` name or `.res` observable it computes the same quantity.",
        "",
        "Adding an observable id to a capability is additive and keeps this version. Changing",
        "the definition behind an existing id bumps `adapter.NORMALIZATION_VERSION` and this",
        "document together.",
        "",
    ]
    for spec in registry.CAPABILITIES.values():
        lines += [
            f"## `{spec.capability}` -- `{spec.sinbad_case}` (`{spec.model}`, {spec.dimension}-D)",
            "",
            "| observable id | definition |",
            "|---|---|",
        ]
        for name, definition in spec.normalization.items():
            lines.append(f"| `{name}` | {definition} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    sys.stdout.write(render_normalization_doc())
