# Known Issues

## Peer Benchmark Pipeline Runtime/Memory Scaling

- File: `similar_banks/src/compute_peer_percentiles.py`
- Current behavior loads and merges the full base dataset in pandas, which is workable for the current dataset size but may become a bottleneck as data volume grows in Databricks driver-only runs.
- Potential solution (not implemented yet): add an option to write a `pb_`-only output (keys + benchmark columns) and perform the final merge in Spark/Databricks.
