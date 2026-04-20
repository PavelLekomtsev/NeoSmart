"""NeoSmart — shared runtime library for the smart parking system.

This package hosts cross-cutting concerns used by both the web app
(`SmartParking/web_app/`) and the training / evaluation pipelines
(`Training/`, `Validation/`):

* `neosmart.config` — typed settings loaded from YAML + env.
* `neosmart.logging_setup` — structured logger factory.
* `neosmart.tracking.sort` — single SORT implementation.
* `neosmart.eval` — detection metrics, threshold sweep, latency.
"""

__version__ = "0.2.0"
