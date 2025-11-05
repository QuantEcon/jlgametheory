# jlgametheory

[![Build Status](https://github.com/QuantEcon/jlgametheory/actions/workflows/ci.yml/badge.svg)](https://github.com/QuantEcon/jlgametheory/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/QuantEcon/jlgametheory/badge.svg)](https://coveralls.io/github/QuantEcon/jlgametheory)
[![Documentation (latest)](https://img.shields.io/badge/docs-latest-blue.svg)](https://quantecon.github.io/jlgametheory/latest/)

Python interface to GameTheory.jl

`jlgametheory` is a Python package that allows passing
a `NormalFormGame` instance from
[`QuantEcon.py`](https://github.com/QuantEcon/QuantEcon.py) to
[`GameTheory.jl`](https://github.com/QuantEcon/GameTheory.jl) functions
via [`JuliaCall`](https://github.com/JuliaPy/PythonCall.jl).
