import numpy as np
from quantecon.game_theory.utilities import NashResult
from . import GameTheory, GameTracer


def to_jl_nfg(g):
    g_jl = GameTheory.NormalFormGame(
        *(GameTheory.Player(player.payoff_array) for player in g.players)
    )
    return g_jl


def _to_py_ne(NE_jl):
    return tuple(x.to_numpy() for x in NE_jl)


def _to_py_nes(NEs_jl):
    return [_to_py_ne(NE) for NE in NEs_jl]


def lrsnash(g):
    """
    Compute in exact arithmetic all extreme mixed-action Nash equilibria
    of a 2-player normal form game with integer payoffs, with the
    lexicographic reverse search vertex enumeration algorithm.

    This function calls the Nash equilibrium computation routine of
    Avis, Rosenberg, Savani, and von Stengel [1]_ implemented in
    `lrslib` (through its Julia wrapper `LRSLib.jl`).

    Parameters
    ----------
    g : NormalFormGame
        2-player NormalFormGame instance with integer payoffs.

    Returns
    -------
    NEs : list(tuple(ndarray(object, ndim=1)))
        List containing tuples of Nash equilibrium mixed actions, where
        the values are represented by `fractions.Fraction`.

    Examples
    --------
    A degenerate game example:

    >>> import quantecon.game_theory as gt
    >>> import jlgametheory as jgt
    >>> from pprint import pprint
    >>> bimatrix = [[(3, 3), (3, 3)],
    ...             [(2, 2), (5, 6)],
    ...             [(0, 3), (6, 1)]]
    >>> g = gt.NormalFormGame(bimatrix)
    >>> NEs = jgt.lrsnash(g)
    >>> pprint(NEs)
    [(array([Fraction(1, 1), Fraction(0, 1), Fraction(0, 1)], dtype=object),
      array([Fraction(1, 1), Fraction(0, 1)], dtype=object)),
     (array([Fraction(1, 1), Fraction(0, 1), Fraction(0, 1)], dtype=object),
      array([Fraction(2, 3), Fraction(1, 3)], dtype=object)),
     (array([Fraction(0, 1), Fraction(1, 3), Fraction(2, 3)], dtype=object),
      array([Fraction(1, 3), Fraction(2, 3)], dtype=object))]

    The set of Nash equilibria of this degenerate game consists of an
    isolated equilibrium, the third output, and a non-singleton
    equilibrium component, the extreme points of which are given by the
    first two outputs.

    References
    ----------
    .. [1] D. Avis, G. Rosenberg, R. Savani, and B. von Stengel,
       "Enumeration of Nash Equilibria for Two-Player Games," Economic
       Theory (2010), 9-37.

    """
    try:
        N = g.N
    except AttributeError:
        raise TypeError('input must be a 2-player NormalFormGame')
    if N != 2:
        raise NotImplementedError('Implemented only for 2-player games')
    if not np.issubdtype(g.dtype, np.integer):
        raise NotImplementedError(
            'Implemented only for games with integer payoffs'
        )

    NEs_jl = GameTheory.lrsnash(to_jl_nfg(g))
    return _to_py_nes(NEs_jl)


def hc_solve(g, ntofind=float('inf'), **options):
    """
    Compute all isolated mixed-action Nash equilibria of an N-player
    normal form game with the polynomial homotopy continuation method.

    This function solves a system of polynomial equations arising from
    the nonlinear complementarity problem representation of Nash
    equilibrium, by using `HomotopyContinuation.jl`.

    Parameters
    ----------
    g : NormalFormGame
        N-player NormalFormGame instance.

    ntofind : scalar, optional(default=float('inf'))
        Number of Nash equilibria to find.

    options :
        Optional keyword arguments to pass to `HomotopyContinuation.solve`.
        For example, the option `seed` can set the random seed used
        during the computations. See the `documentation
        <https://www.juliahomotopycontinuation.org/HomotopyContinuation.jl/stable/solve/>`_
        for `HomotopyContinuation.solve` for details.

    Returns
    -------
    NEs : list(tuple(ndarray(float, ndim=1)))
        List containing tuples of Nash equilibrium mixed actions.

    Examples
    --------
    Consider the 3-player 2-action game with 9 Nash equilibria in
    McKelvey and McLennan (1996) "Computation of Equilibria in Finite
    Games":

    >>> import quantecon.game_theory as gt
    >>> import jlgametheory as jgt
    >>> from pprint import pprint
    >>> import numpy as np
    >>> np.set_printoptions(precision=3)  # Reduce the digits printed
    >>> g = gt.NormalFormGame((2, 2, 2))
    >>> g[0, 0, 0] = 9, 8, 12
    >>> g[1, 1, 0] = 9, 8, 2
    >>> g[0, 1, 1] = 3, 4, 6
    >>> g[1, 0, 1] = 3, 4, 4
    >>> print(g)
    3-player NormalFormGame with payoff profile array:
    [[[[ 9.,  8., 12.],   [ 0.,  0.,  0.]],
      [[ 0.,  0.,  0.],   [ 3.,  4.,  6.]]],
    <BLANKLINE>
     [[[ 0.,  0.,  0.],   [ 3.,  4.,  4.]],
      [[ 9.,  8.,  2.],   [ 0.,  0.,  0.]]]]
    >>> NEs = jgt.hc_solve(g, show_progress=False)
    >>> len(NEs)
    9
    >>> pprint(NEs)
    [(array([0., 1.]), array([0., 1.]), array([1., 0.])),
     (array([0.5, 0.5]), array([0.5, 0.5]), array([1.000e+00, 2.351e-38])),
     (array([1., 0.]), array([0., 1.]), array([-1.881e-37,  1.000e+00])),
     (array([0.25, 0.75]), array([0.5, 0.5]), array([0.333, 0.667])),
     (array([0.25, 0.75]), array([1.000e+00, 1.345e-43]), array([0.25, 0.75])),
     (array([0., 1.]), array([0.333, 0.667]), array([0.333, 0.667])),
     (array([1., 0.]), array([ 1.00e+00, -5.74e-42]), array([1., 0.])),
     (array([0., 1.]), array([1., 0.]), array([2.374e-66, 1.000e+00])),
     (array([0.5, 0.5]), array([0.333, 0.667]), array([0.25, 0.75]))]
    >>> all(g.is_nash(NE) for NE in NEs)
    True

    """
    try:
        N = g.N
    except AttributeError:
        raise TypeError('g must be a NormalFormGame')
    if N < 2:
        raise NotImplementedError('Not implemented for 1-player games')

    NEs_jl = GameTheory.hc_solve(to_jl_nfg(g), ntofind=ntofind, **options)
    return _to_py_nes(NEs_jl)


def _make_ray(ray, M, rng):
    """
    Return `ray` as a float64 ndarray of shape (M,), or generate one
    randomly with `rng` if `ray` is None.

    """
    if ray is None:
        return np.random.default_rng(rng).random(M)
    ray = np.asarray(ray, dtype=np.float64)
    if ray.shape != (M,):
        raise ValueError(
            'ray must be a 1-dim array of length sum(g.nums_actions)'
        )
    return ray


def ipa_solve(g, *, ray=None, rng=None, full_output=False, **options):
    """
    Compute one mixed-action approximate Nash equilibrium of an N-player
    normal form game with the iterated polymatrix approximation (IPA)
    algorithm.

    This function calls the IPA routine of Govindan and Wilson [1]_ from
    the C++ library `GameTracer
    <http://dags.stanford.edu/Games/gametracer.html>`_ through its Julia
    wrapper `GameTracer.jl
    <https://github.com/QuantEcon/GameTracer.jl>`_.

    Parameters
    ----------
    g : NormalFormGame
        NormalFormGame instance with N >= 2 players.

    ray : array_like(float, ndim=1), optional(default=None)
        Perturbation ray for the IPA homotopy, of length
        `sum(g.nums_actions)`, where the entries are interpreted in
        player-concatenated order. Different rays may lead to different
        equilibria. If None, a ray is randomly generated by `rng`.

    rng : int or np.random.Generator, optional(default=None)
        Random seed or random number generator used to generate the
        default `ray` (passed to `np.random.default_rng`). Irrelevant if
        `ray` is given.

    full_output : bool, optional(default=False)
        If False, only the computed Nash equilibrium is returned. If
        True, the return value is `(NE, res)`, where `res` is a
        `NashResult` object.

    options :
        Optional keyword arguments to pass to `GameTracer.ipa_solve`:
        `zh_init` (initial condition for z-hat, default all ones),
        `alpha` (step size fraction, default 0.02), `fuzz` (stopping
        tolerance, default 1e-6), `max_iter` (maximum number of
        iterations, i.e., polymatrix approximations, default 100000),
        and `max_pivots` (maximum number of pivoting steps per
        Lemke-Howson solve, default 1000000). See the `GameTracer.jl
        documentation
        <https://quantecon.github.io/GameTracer.jl/stable/#GameTracer.ipa_solve>`_
        for details.

    Returns
    -------
    NE : tuple(ndarray(float, ndim=1))
        Tuple of computed (approximate) Nash equilibrium mixed actions.
        If the routine did not converge (see `res.converged`), the last
        iterate is returned.

    res : NashResult
        Object containing the information about the result, returned
        only when `full_output` is True, with the following attributes:
        `NE` (same as `NE` above), `converged` (whether an
        equilibrium was found within `max_iter` iterations without the
        routine giving up), `ret_code` (return code from the IPA
        routine, 1 on success and 0 otherwise), `num_iter` (number of
        iterations performed), `max_iter` (maximum number of
        iterations), and `ray` (perturbation ray used).

    Examples
    --------
    Consider the 3-player 2-action game with 9 Nash equilibria in
    McKelvey and McLennan (1996) "Computation of Equilibria in Finite
    Games":

    >>> import numpy as np
    >>> import quantecon.game_theory as gt
    >>> import jlgametheory as jgt
    >>> np.set_printoptions(precision=3)  # Reduce the digits printed
    >>> rng = np.random.default_rng(0)
    >>> g = gt.NormalFormGame((2, 2, 2))
    >>> g[0, 0, 0] = 9, 8, 12
    >>> g[1, 1, 0] = 9, 8, 2
    >>> g[0, 1, 1] = 3, 4, 6
    >>> g[1, 0, 1] = 3, 4, 4
    >>> NE = jgt.ipa_solve(g, rng=rng)
    >>> NE
    (array([1., 0.]), array([1., 0.]), array([1., 0.]))
    >>> g.is_nash(NE)
    True

    Calls with different rays generally yield different equilibria
    (here, `rng` generates a different ray on each call as its state
    advances):

    >>> NE = jgt.ipa_solve(g, rng=rng)
    >>> NE
    (array([0.25, 0.75]), array([0.5, 0.5]), array([0.333, 0.667]))
    >>> g.is_nash(NE, tol=1e-5)  # Relax tolerance
    True

    References
    ----------
    .. [1] S. Govindan and R. Wilson, "Computing Nash Equilibria by
       Iterated Polymatrix Approximation," Journal of Economic Dynamics
       and Control, 28 (2004), 1229-1241.

    """
    try:
        N = g.N
    except AttributeError:
        raise TypeError('g must be a NormalFormGame')
    if N < 2:
        raise NotImplementedError('Not implemented for 1-player games')

    ray = _make_ray(ray, sum(g.nums_actions), rng)
    res_jl = GameTracer.ipa_solve(to_jl_nfg(g), ray=ray, **options)
    NE = _to_py_ne(res_jl.NE)

    if not full_output:
        return NE
    res = NashResult(NE=NE,
                     converged=bool(res_jl.converged),
                     ret_code=int(res_jl.ret_code),
                     num_iter=int(res_jl.num_iter),
                     max_iter=int(res_jl.max_iter),
                     ray=res_jl.ray.to_numpy())
    return NE, res


def gnm_solve(g, *, ray=None, rng=None, full_output=False, **options):
    """
    Compute mixed-action Nash equilibria of an N-player normal form game
    with the global Newton method (GNM) algorithm.

    This function calls the GNM routine of Govindan and Wilson [1]_ from
    the C++ library `GameTracer
    <http://dags.stanford.edu/Games/gametracer.html>`_ through its Julia
    wrapper `GameTracer.jl
    <https://github.com/QuantEcon/GameTracer.jl>`_.

    Parameters
    ----------
    g : NormalFormGame
        NormalFormGame instance with N >= 2 players.

    ray : array_like(float, ndim=1), optional(default=None)
        Perturbation ray for the GNM homotopy, of length
        `sum(g.nums_actions)`, where the entries are interpreted in
        player-concatenated order. Different rays may lead to different
        sets of equilibria. If None, a ray is randomly generated by
        `rng`.

    rng : int or np.random.Generator, optional(default=None)
        Random seed or random number generator used to generate the
        default `ray` (passed to `np.random.default_rng`). Irrelevant if
        `ray` is given.

    full_output : bool, optional(default=False)
        If False, only the list of computed Nash equilibria is returned.
        If True, the return value is `(NEs, res)`, where `res` is a
        `NashResult` object.

    options :
        Optional keyword arguments to pass to `GameTracer.gnm_solve`:
        `steps` (number of steps within a support cell, default 100),
        `fuzz` (numerical zero threshold, default 1e-12), `lnmfreq`
        (frequency of local Newton method corrections, default 3),
        `lnmmax` (maximum iterations within the local Newton method,
        default 10), `lambdamin` (minimum value of the continuation
        parameter, default -10.0), `wobble` (whether to use "wobbles" of
        the perturbation vector, default False), `threshold` (error
        threshold to trigger a wobble, default 1e-2), and `max_iter`
        (maximum number of iterations, i.e., support cells traversed,
        default 5000; if reached, the equilibria found so far are
        returned). See the `GameTracer.jl documentation
        <https://quantecon.github.io/GameTracer.jl/stable/#GameTracer.gnm_solve>`_
        for details.

    Returns
    -------
    NEs : list(tuple(ndarray(float, ndim=1)))
        List containing tuples of computed Nash equilibrium mixed
        actions. If `max_iter` is reached, the list contains the
        equilibria found up to that point.

    res : NashResult
        Object containing the information about the result, returned
        only when `full_output` is True, with the following attributes:
        `NEs` (same as `NEs` above), `ret_code` (return code from
        the GNM routine, the number of equilibria found), `num_iter`
        (number of iterations performed), `max_iter` (maximum number of
        iterations), and `ray` (perturbation ray used).

    Examples
    --------
    Consider the 3-player 2-action game with 9 Nash equilibria in
    McKelvey and McLennan (1996) "Computation of Equilibria in Finite
    Games":

    >>> import numpy as np
    >>> import quantecon.game_theory as gt
    >>> import jlgametheory as jgt
    >>> from pprint import pprint
    >>> np.set_printoptions(precision=3)  # Reduce the digits printed
    >>> rng = np.random.default_rng(28)
    >>> g = gt.NormalFormGame((2, 2, 2))
    >>> g[0, 0, 0] = 9, 8, 12
    >>> g[1, 1, 0] = 9, 8, 2
    >>> g[0, 1, 1] = 3, 4, 6
    >>> g[1, 0, 1] = 3, 4, 4
    >>> NEs = jgt.gnm_solve(g, rng=rng)
    >>> pprint(NEs)
    [(array([0., 1.]), array([1., 0.]), array([0., 1.])),
     (array([0.25, 0.75]), array([1., 0.]), array([0.25, 0.75])),
     (array([1., 0.]), array([1., 0.]), array([1., 0.]))]

    Calls with different rays generally yield different sets of
    equilibria (here, `rng` generates a different ray on each call as
    its state advances):

    >>> NEs = jgt.gnm_solve(g, rng=rng)
    >>> pprint(NEs)
    [(array([1., 0.]), array([0., 1.]), array([0., 1.])),
     (array([0.5, 0.5]), array([0.333, 0.667]), array([0.25, 0.75])),
     (array([0., 1.]), array([1., 0.]), array([0., 1.])),
     (array([0.25, 0.75]), array([1., 0.]), array([0.25, 0.75])),
     (array([0.25, 0.75]), array([0.5, 0.5]), array([0.333, 0.667])),
     (array([0., 1.]), array([0.333, 0.667]), array([0.333, 0.667])),
     (array([0., 1.]), array([0., 1.]), array([1., 0.])),
     (array([0.5, 0.5]), array([0.5, 0.5]), array([1., 0.])),
     (array([1., 0.]), array([1., 0.]), array([1., 0.]))]
    >>> all(g.is_nash(NE, tol=1e-5) for NE in NEs)
    True

    References
    ----------
    .. [1] S. Govindan and R. Wilson, "A Global Newton Method to Compute
       Nash Equilibria," Journal of Economic Theory, 110 (2003), 65-86.

    """
    try:
        N = g.N
    except AttributeError:
        raise TypeError('g must be a NormalFormGame')
    if N < 2:
        raise NotImplementedError('Not implemented for 1-player games')

    ray = _make_ray(ray, sum(g.nums_actions), rng)
    res_jl = GameTracer.gnm_solve(to_jl_nfg(g), ray=ray, **options)
    NEs = _to_py_nes(res_jl.NEs)

    if not full_output:
        return NEs
    res = NashResult(NEs=NEs,
                     ret_code=int(res_jl.ret_code),
                     num_iter=int(res_jl.num_iter),
                     max_iter=int(res_jl.max_iter),
                     ray=res_jl.ray.to_numpy())
    return NEs, res
