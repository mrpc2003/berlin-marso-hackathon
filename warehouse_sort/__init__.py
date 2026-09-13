"""Warehouse Colour-Sort hackathon starter package.

Importing this package registers the WarehouseSort-v1 ManiSkill environment. The registration is
guarded so that the policy loaders (`warehouse_sort.il_policy`) stay importable on machines without
ManiSkill/SAPIEN (e.g. for CPU unit tests); eval.py / the judge always run with ManiSkill installed.
"""

try:
    from warehouse_sort.env import WarehouseSortEnv  # noqa: F401  (registers WarehouseSort-v1)
except ImportError as _e:  # pragma: no cover - only on machines without the simulator
    import warnings

    warnings.warn(f"warehouse_sort.env not imported (simulator missing?): {_e}")
    WarehouseSortEnv = None  # type: ignore

__all__ = ["WarehouseSortEnv"]
