"""sinbad-oracle-fenicsx: independent FEniCSx/dolfinx verification oracle.

Implements the `sinbad-oracle-protocol/1` adapter contract (SV0-C5) for
Sinbad's SV0 trustworthy-simulation-factory oracle tier. See README.md for
scope and INSTALL.md for the dolfinx environment this adapter needs.
"""

from .protocol import (
    ORACLE_PROTOCOL_SCHEMA,
    ORACLE_REQUEST_SCHEMA,
    ORACLE_RESULT_SCHEMA,
    OracleRequest,
    OracleResult,
    OracleToolIdentity,
    ProtocolError,
)

__all__ = [
    "ORACLE_PROTOCOL_SCHEMA",
    "ORACLE_REQUEST_SCHEMA",
    "ORACLE_RESULT_SCHEMA",
    "OracleRequest",
    "OracleResult",
    "OracleToolIdentity",
    "ProtocolError",
]
