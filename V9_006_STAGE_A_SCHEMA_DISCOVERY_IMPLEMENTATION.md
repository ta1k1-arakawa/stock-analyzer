# Stage-A schema discovery implementation

`src/v9_006_stage_a_schema_discovery.py` provides deterministic offline
profile generation from synthetic or future verified locks. It records safe
provenance, format, byte length, hash, and a value-free structural hash. It
does not bind parser mappings or execute acquisition.
