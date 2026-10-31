"""
jlgametheory: Python interface to GameTheory.jl

"""
from juliacall import Main as jl
from importlib.metadata import PackageNotFoundError, version as _version

jl.seval("using GameTheory")
GameTheory = jl.GameTheory
jl.seval("using GameTracer")
GameTracer = jl.GameTracer
jl.seval("using StrategicGames")
StrategicGames = jl.StrategicGames

from .jlgametheory import (
    lrsnash, support_enumeration, hc_solve, ipa_solve, gnm_solve,
    nash_cp, nash_se
)

try:
    __version__ = _version("jlgametheory")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"

__all__ = [
    "lrsnash", "support_enumeration", "hc_solve", "ipa_solve", "gnm_solve",
    "nash_cp", "nash_se"
]
