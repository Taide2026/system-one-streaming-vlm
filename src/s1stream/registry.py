"""Name -> plugin class registries, one per slot. A config spec {"name": ..., **params} builds a plugin."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


class RegistryError(Exception):
    """Unknown or duplicate plugin name. The message names the slot and the known names."""


class Registry:
    """Holds the plugin classes of one slot."""

    def __init__(self, slot: str) -> None:
        self.slot = slot
        self._classes: dict[str, type] = {}

    def register(self, name: str) -> Callable[[type], type]:
        """Class decorator. Add the class under name and return the class unchanged.

        Raise RegistryError when name is already registered in this slot.
        """
        def decorator(cls: type) -> type:
            if name in self._classes:
                raise RegistryError(f"Duplicate {self.slot} plugin {name!r}; known names: {', '.join(self.names())}")
            self._classes[name] = cls
            return cls

        return decorator

    def names(self) -> list[str]:
        """Return the registered names, sorted."""
        return sorted(self._classes)

    def build(self, spec: Mapping[str, Any]) -> Any:
        """Create the plugin spec["name"] with the other keys of spec as keyword arguments.

        Raise RegistryError when spec has no "name" or the name is unknown.
        The message of an unknown name lists the known names.
        """
        name = spec.get("name")
        if name not in self._classes:
            raise RegistryError(
                f"Unknown {self.slot} plugin {name!r}; known names: {', '.join(self.names())}"
            )
        params = {key: value for key, value in spec.items() if key != "name"}
        return self._classes[name](**params)


DATASETS = Registry("dataset")
INPUT_FORMS = Registry("input_form")
BACKBONES = Registry("backbone")
ANSWER_HEADS = Registry("answer_head")
DECISION_RULES = Registry("decision_rule")
