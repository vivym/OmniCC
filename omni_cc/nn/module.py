from typing import Iterable

from tvm import relax
from tvm.relax.testing.nn import (
    Module as _Module,
    Parameter
)


class Module(_Module):
    def named_parameters(self) -> dict[str, Parameter]:
        params: dict[str, Parameter] = {}
        for name, module in self.__dict__.items():
            if isinstance(module, Parameter):
                params[name] = module
            elif isinstance(module, ModuleList):
                for i, m in enumerate(module):
                    for param_name, param in m.named_parameters().items():
                        params[f"{name}.{i}.{param_name}"] = param
            elif isinstance(module, Module):
                for param_name, param in module.named_parameters().items():
                    params[f"{name}.{param_name}"] = param

        return params


class ModuleList(Module):
    def __init__(self, modules: list[Module] | None = None):
        if modules is None:
            modules = []

        for m in modules:
            assert isinstance(m, Module)

        self.modules = modules

    def append(self, module: Module) -> None:
        assert isinstance(module, Module)
        self.modules.append(module)

    def __iter__(self) -> Iterable[Module]:
        return iter(self.modules)

    def __getitem__(self, idx) -> Module:
        return self.modules[idx]

    def __len__(self) -> int:
        return len(self.modules)

    def forward(self, x: relax.Expr) -> relax.Var:
        for module in self.modules:
            x = module(x)
        return x
