jlgametheory — Documentation
============================

`jlgametheory <https://github.com/QuantEcon/jlgametheory>`_ is a Python interface to
`GameTheory.jl <https://github.com/QuantEcon/GameTheory.jl>`_.

It allows passing a `NormalFormGame` instance from
`quantecon.game_theory <https://quanteconpy.readthedocs.io/en/stable/game_theory.html>`_
to GameTheory.jl functions via
`juliacall <https://juliapy.github.io/PythonCall.jl/stable/juliacall/>`_.

* For constructing `NormalFormGame`,
  see its `documentation <https://quanteconpy.readthedocs.io/en/stable/game_theory/normal_form_game.html>`_.


Implemented functions
---------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   :template: function

   ~jlgametheory.lrsnash
   ~jlgametheory.hc_solve
