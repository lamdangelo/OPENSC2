"""Observer composition."""

from __future__ import annotations

from typing import Iterable

from opensc2.ports.observer import NullObserver, SimulationObserver


class CompositeObserver(NullObserver):
    """Forwards every hook to its observers, in registration order. The
    order matters only for console output; every observer writes its own
    files."""

    def __init__(self, observers: Iterable[SimulationObserver] = ()):
        self.observers = list(observers)

    def add(self, observer: SimulationObserver) -> None:
        self.observers.append(observer)

    def __getattribute__(self, name):
        if name.startswith(("on_", "after_")):
            observers = object.__getattribute__(self, "observers")

            def forward(*args, **kwargs):
                for observer in observers:
                    getattr(observer, name)(*args, **kwargs)

            return forward
        return object.__getattribute__(self, name)
