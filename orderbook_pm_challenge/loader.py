from __future__ import annotations

import importlib.util
import pathlib
import types


def load_strategy_factory(strategy_path: str):
    path = pathlib.Path(strategy_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Strategy file does not exist: {path}")

    module_name = f"orderbook_pm_strategy_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load strategy module from {path}")

    module = importlib.util.module_from_spec(spec)
    assert isinstance(module, types.ModuleType)
    spec.loader.exec_module(module)

    strategy_cls = getattr(module, "Strategy", None)
    if strategy_cls is None:
        raise AttributeError(f"{path} does not define a Strategy class")

    def factory():
        instance = strategy_cls()
        if not hasattr(instance, "on_step"):
            raise TypeError("Strategy instance must define on_step(state)")
        return instance

    return factory
