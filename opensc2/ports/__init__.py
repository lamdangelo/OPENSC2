"""Ports: Protocols that the domain defines and the outer layers implement.

Dependencies point inward only (docs/architecture_refactoring_plan.md): the
domain and application layers import these Protocols, the adapters layer
provides the implementations, and the composition root (simulation.py)
wires them together.
"""
