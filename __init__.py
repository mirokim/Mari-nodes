# -*- coding: utf-8 -*-
"""
Mari Nodes package initializer for ComfyUI

- Collects NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS
  from all submodules under this package and exposes them at
  the package level (this file), so ComfyUI can register nodes reliably.
"""

from __future__ import annotations
import importlib
import pkgutil
import sys
import types
from typing import Dict, Any

__version__ = "1.2.0"  # bump as you update the package

# =========================================================
# Package-level registries (ComfyUI reads these)
# =========================================================
NODE_CLASS_MAPPINGS: Dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _merge_mapping(dst: dict, src: dict, src_mod_name: str, kind: str) -> None:
    """Merge src dict into dst with duplicate-key warning."""
    for k, v in src.items():
        if k in dst and dst[k] is not v:
            print(f"[Mari Nodes] Warning: duplicate {kind} key '{k}' "
                  f"overwritten by module '{src_mod_name}'")
        dst[k] = v

def _merge_module(mod: types.ModuleType) -> None:
    """Pick up node mappings from a module if present (ComfyUI style)."""
    # Standard ComfyUI keys
    if hasattr(mod, "NODE_CLASS_MAPPINGS") and isinstance(mod.NODE_CLASS_MAPPINGS, dict):
        _merge_mapping(NODE_CLASS_MAPPINGS, mod.NODE_CLASS_MAPPINGS, mod.__name__, "NODE_CLASS_MAPPINGS")
    # Optional pretty display names
    if hasattr(mod, "NODE_DISPLAY_NAME_MAPPINGS") and isinstance(mod.NODE_DISPLAY_NAME_MAPPINGS, dict):
        _merge_mapping(NODE_DISPLAY_NAME_MAPPINGS, mod.NODE_DISPLAY_NAME_MAPPINGS, mod.__name__, "NODE_DISPLAY_NAME_MAPPINGS")
    # Legacy compatibility (rare)
    if hasattr(mod, "NODE_MAPPINGS") and isinstance(mod.NODE_MAPPINGS, dict):
        _merge_mapping(NODE_CLASS_MAPPINGS, mod.NODE_MAPPINGS, mod.__name__, "NODE_MAPPINGS(legacy)")

def _safe_import(fullname: str) -> types.ModuleType | None:
    try:
        return importlib.import_module(fullname)
    except Exception as e:
        print(f"[Mari Nodes] Error importing '{fullname}': {e}")
        return None

# =========================================================
# Import submodules and merge their mappings
# =========================================================
# NOTE:
#  - We intentionally do NOT import submodules at the top of file
#    to avoid early reference / circular import issues.
#  - We import them here, after registries are defined.

if "__path__" in globals():
    # Auto-import every .py module inside this package
    for _m in pkgutil.iter_modules(__path__):
        # Skip private/dunder modules just in case
        if _m.name.startswith("_"):
            continue
        mod = _safe_import(f"{__name__}.{_m.name}")
        if mod is not None:
            _merge_module(mod)
else:
    # Fallback: if somehow loaded as a single module (not a package)
    # try to merge from ourselves only. This is unusual for ComfyUI.
    _merge_module(sys.modules.get(__name__))

# =========================================================
# Finalization
# =========================================================
# Optional: provide __all__ for cleaner wildcard imports
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "__version__",
]

print(f"[Mari Nodes] OK v{__version__} — registered {len(NODE_CLASS_MAPPINGS)} classes, "
      f"{len(NODE_DISPLAY_NAME_MAPPINGS)} display names")
