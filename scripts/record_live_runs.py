"""Records live adapter runs as offline fixtures (run inside the dolfinx image).

    scripts/dolfinx-image.sh python3 scripts/record_live_runs.py tests/fixtures/recorded

For every capability and every level of the mirrored Sinbad case's own
refinement ladder, invokes the real CLI (`python -m sinbad_oracle_fenicsx
request result`) exactly as Sinbad's harness does, requesting every
registered observable, and copies request.json, result.json and (for a
satisfied run) the evidence manifest into `<out>/<capability>/<nx>x<ny>/`.
The .npz mesh/solution arrays stay out of the repository; their sha256
digests in the manifest identify them.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from sinbad_oracle_fenicsx import evidence, protocol, registry
from sinbad_oracle_fenicsx.adapter import _actual_tool_identity

# The refinement ladders the Sinbad case files author (3-D ladders as [n, n]).
LADDERS = {
    "poisson": [(4, 4), (8, 8), (16, 16)],
    "nonlinear_heat": [(2, 2), (4, 4), (8, 8)],
    "linear_elasticity": [(2, 2), (4, 4)],
    "stokes": [(2, 2), (4, 4), (8, 8)],
    "mixed_darcy": [(2, 2)],
}


def main(out_root: Path) -> int:
    tool = _actual_tool_identity()
    if tool.version == "unavailable":
        print("dolfinx unavailable; nothing recorded", file=sys.stderr)
        return 1
    for capability, ladder in LADDERS.items():
        spec = registry.CAPABILITIES[capability]
        for nx, ny in ladder:
            with tempfile.TemporaryDirectory() as tmp:
                io = Path(tmp)
                request = {
                    "schema": protocol.ORACLE_REQUEST_SCHEMA,
                    "tool": tool.to_dict(),
                    "capability": capability,
                    "case_id": f"{spec.sinbad_case}/recorded-{nx}x{ny}",
                    "model_digest": "blake3:recorded-fixture",
                    "refinement": [nx, ny],
                    "observables": sorted(spec.observables),
                }
                (io / "request.json").write_text(json.dumps(request, indent=2))
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "sinbad_oracle_fenicsx",
                        io / "request.json",
                        io / "result.json",
                    ],
                    capture_output=True,
                    text=True,
                )
                if completed.returncode != 0:
                    print(f"{capability} {nx}x{ny}: crash\n{completed.stderr}", file=sys.stderr)
                    return 1
                target = out_root / capability / f"{nx}x{ny}"
                if target.exists():
                    shutil.rmtree(target)
                target.mkdir(parents=True)
                shutil.copy(io / "request.json", target / "request.json")
                shutil.copy(io / "result.json", target / "result.json")
                manifest = (
                    evidence.evidence_dir_for(io / "result.json") / evidence.MANIFEST_NAME
                )
                result = json.loads((target / "result.json").read_text())
                if manifest.exists():
                    shutil.copy(manifest, target / "manifest.json")
                    values = {
                        k: protocol.bits_to_finite_f64(v)
                        for k, v in result["observables"].items()
                    }
                    print(f"{capability} {nx}x{ny}: satisfied {json.dumps(values)}")
                else:
                    print(f"{capability} {nx}x{ny}: {json.dumps(result['status'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
