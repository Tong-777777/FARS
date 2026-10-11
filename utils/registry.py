import sys
from typing import Callable, Optional, Type, Dict, List

class Registry:
    """Registry to map string names to classes, allowing for dynamic class retrieval.

    Args:
        name (str): The registry's name.
        module_key (str, optional): Module name key to update its __all__ variable if needed.
            Defaults to None.
    """

    def __init__(self, name: str, module_key: Optional[str] = None) -> None:
        self.name = name
        self.module_key = module_key
        self._registered_objects: Dict[str, Type] = {}

    def register(self) -> Callable[[Type], Type]:
        """Decorator to register a class in the registry by its name."""

        def _register(cls: Type) -> Type:
            self._registered_objects[cls.__name__] = cls
            if self.module_key:
                module = sys.modules.get(self.module_key)
                if module and hasattr(module, "__all__"):
                    module.__all__.append(cls.__name__)
            return cls

        return _register

    @property
    def registry_names(self) -> List[str]:
        """List of all registered class names."""
        return list(self._registered_objects.keys())

    def __getitem__(self, key: str) -> Type:
        """Retrieve a class by its name."""
        if key not in self._registered_objects:
            raise KeyError(f"Class '{key}' not found in registry '{self.name}'.")
        return self._registered_objects[key]

    def __len__(self) -> int:
        """Return the number of registered classes."""
        return len(self._registered_objects)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', " \
               f"registered_objects={list(self._registered_objects.keys())})"
