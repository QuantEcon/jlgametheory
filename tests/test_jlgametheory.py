from fractions import Fraction
import numpy as np
from numpy.testing import assert_, assert_raises
from quantecon.game_theory import NormalFormGame
from jlgametheory import (
    lrsnash, support_enumeration, hc_solve, ipa_solve, gnm_solve, nash_cp,
    nash_se
)


def compare_act_profs(operator, act_prof1, act_prof2, *args, **kwargs):
    if len(act_prof1) != len(act_prof2):
        return False
    for a1, a2 in zip(act_prof1, act_prof2):
        if not operator(a1, a2, *args, **kwargs):
            return False
    return True


def compare_lists_act_profs(operator, list_act_profs1, list_act_profs2,
                            *args, **kwargs):
    if len(list_act_profs1) != len(list_act_profs2):
        return False
    unmatched = list(list_act_profs1)
    for prof2 in list_act_profs2:
        for i, prof1 in enumerate(unmatched):
            if compare_act_profs(operator, prof1, prof2, *args, **kwargs):
                unmatched.pop(i)
                break
        else:
            return False
    return True


class TestLRSNash:
    def setup_method(self):
        self.game_dicts = []

        # From von Stengel 2007 in Algorithmic Game Theory
        bimatrix = [[(3, 3), (3, 2)],
                    [(2, 2), (5, 6)],
                    [(0, 3), (6, 1)]]
        d = {'g': NormalFormGame(bimatrix),
             'NEs': [([Fraction(1), Fraction(0), Fraction(0)],
                      [Fraction(1), Fraction(0)]),
                     ([Fraction(4, 5), Fraction(1, 5), Fraction(0)],
                      [Fraction(2, 3), Fraction(1, 3)]),
                     ([Fraction(0), Fraction(1, 3), Fraction(2, 3)],
                      [Fraction(1, 3), Fraction(2, 3)])]}
        self.game_dicts.append(d)

        # Degenerate game
        bimatrix = [[(3, 3), (3, 3)],
                    [(2, 2), (5, 6)],
                    [(0, 3), (6, 1)]]
        d = {'g': NormalFormGame(bimatrix),
             'NEs': [([Fraction(1), Fraction(0), Fraction(0)],
                      [Fraction(1), Fraction(0)]),
                     ([Fraction(1), Fraction(0), Fraction(0)],
                      [Fraction(2, 3), Fraction(1, 3)]),
                     ([Fraction(0), Fraction(1, 3), Fraction(2, 3)],
                      [Fraction(1, 3), Fraction(2, 3)])]}
        self.game_dicts.append(d)

    def test_lrsnash(self):
        for d in self.game_dicts:
            NEs_computed = lrsnash(d['g'])
            assert_(compare_lists_act_profs(
                np.array_equal, NEs_computed, d['NEs']
            )
            )

    def test_invalid_non_integer_g(self):
        g_float = NormalFormGame(
            self.game_dicts[0]['g'].payoff_profile_array.astype('float')
        )
        assert_raises(NotImplementedError, lrsnash, g_float)


def _three_player_game_dicts():
    game_dicts = []

    # 2x2x2 game from McKelvey and McLennan
    g = NormalFormGame((2, 2, 2))
    g[0, 0, 0] = 9, 8, 12
    g[1, 1, 0] = 9, 8, 2
    g[0, 1, 1] = 3, 4, 6
    g[1, 0, 1] = 3, 4, 4
    NEs = [
        ([1, 0], [1, 0], [1, 0]),
        ([0, 1], [0, 1], [1, 0]),
        ([1, 0], [0, 1], [0, 1]),
        ([0, 1], [1, 0], [0, 1]),
        ([0, 1], [1/3, 2/3], [1/3, 2/3]),
        ([1/4, 3/4], [1, 0], [1/4, 3/4]),
        ([1/2, 1/2], [1/2, 1/2], [1, 0]),
        ([1/4, 3/4], [1/2, 1/2], [1/3, 2/3]),
        ([1/2, 1/2], [1/3, 2/3], [1/4, 3/4])
    ]
    game_dicts.append({'g': g, 'NEs': NEs})

    # 2x2x2 game from Nau, Canovas, and Hansen
    payoff_profile_array = [
        [[(3, 0, 2), (1, 0, 0)],
         [(0, 2, 0), (0, 1, 0)]],
        [[(0, 1, 0), (0, 3, 0)],
         [(1, 0, 0), (2, 0, 3)]]
    ]
    q = (-13 + np.sqrt(601)) / 24
    p = (9*q - 1) / (7*q + 2)
    r = (-3*q + 2) / (q + 1)
    game_dicts.append({'g': NormalFormGame(payoff_profile_array),
                       'NEs': [([p, 1-p], [q, 1-q], [r, 1-r])]})

    return game_dicts


class TestSupportEnumeration:
    def setup_method(self):
        self.game_dicts = _three_player_game_dicts()

    def test_support_enumeration(self):
        for d in self.game_dicts:
            NEs_computed = support_enumeration(d['g'])
            assert_(compare_lists_act_profs(
                np.allclose, NEs_computed, d['NEs'], atol=1e-8
            )
            )

    def test_ordered_by_support_size(self):
        g = self.game_dicts[0]['g']
        NEs = support_enumeration(g)
        sizes = [sum((x > 0).sum() for x in NE) for NE in NEs]
        assert_(sizes == sorted(sizes))

    def test_ntofind(self):
        # 9 equilibria: 4 pure, then mixed ones; 8 stops within the
        # solutions of the fully mixed support
        g = self.game_dicts[0]['g']
        for ntofind in [0, 1, 3, 5, 8]:
            NEs = support_enumeration(g, ntofind=ntofind)
            assert_(len(NEs) == ntofind)
            for NE in NEs:
                assert_(g.is_nash(NE))

    def test_tol(self):
        # Player 0's action 0 is worse than action 1 by 5e-7, so that
        # the pure profile (0, 0, 0) is an equilibrium only within the
        # looser tolerance
        g = NormalFormGame((2, 1, 1))
        g[1, 0, 0] = 5e-7, 0, 0
        assert_(len(support_enumeration(g, tol=1e-8)) == 1)
        assert_(len(support_enumeration(g, tol=1e-6)) == 2)

    def test_seed_reproducibility(self):
        g = self.game_dicts[0]['g']
        NEs0 = support_enumeration(g, seed=1234)
        NEs1 = support_enumeration(g, seed=1234)
        assert_(compare_lists_act_profs(np.array_equal, NEs0, NEs1))

    def test_max_retries(self):
        d = self.game_dicts[0]
        NEs_computed = support_enumeration(d['g'], max_retries=0, seed=1234)
        assert_(compare_lists_act_profs(
            np.allclose, NEs_computed, d['NEs'], atol=1e-8
        )
        )

    def test_invalid_2player_g(self):
        bimatrix = [[(3, 3), (3, 2)],
                    [(2, 2), (5, 6)],
                    [(0, 3), (6, 1)]]
        assert_raises(NotImplementedError, support_enumeration,
                      NormalFormGame(bimatrix))


class TestHCSolve:
    def setup_method(self):
        # From von Stengel 2007 in Algorithmic Game Theory
        bimatrix = [[(3, 3), (3, 2)],
                    [(2, 2), (5, 6)],
                    [(0, 3), (6, 1)]]
        d = {'g': NormalFormGame(bimatrix),
             'NEs': [([1, 0, 0], [1, 0]),
                     ([4/5, 1/5, 0], [2/3, 1/3]),
                     ([0, 1/3, 2/3], [1/3, 2/3])]}
        self.game_dicts = [d] + _three_player_game_dicts()

    def test_hc_solve(self):
        for d in self.game_dicts:
            NEs_computed = hc_solve(d['g'], show_progress=False)
            assert_(compare_lists_act_profs(
                np.allclose, NEs_computed, d['NEs'], atol=1e-15
            )
            )

    def test_ntofind(self):
        g = self.game_dicts[1]['g']
        NEs_computed = hc_solve(g, ntofind=1, show_progress=False)
        assert_(len(NEs_computed) == 1)
        assert_(g.is_nash(NEs_computed[0]))

    def test_seed(self):
        d = self.game_dicts[0]
        NEs_computed = hc_solve(d['g'], seed=1234, show_progress=False)
        assert_(compare_lists_act_profs(
            np.allclose, NEs_computed, d['NEs'], atol=1e-15
        )
        )


def _gametracer_game_dicts():
    game_dicts = []

    # From von Stengel 2007 in Algorithmic Game Theory
    bimatrix = [[(3, 3), (3, 2)],
                [(2, 2), (5, 6)],
                [(0, 3), (6, 1)]]
    game_dicts.append({'g': NormalFormGame(bimatrix)})

    # 2x2x2 game from McKelvey and McLennan
    g = NormalFormGame((2, 2, 2))
    g[0, 0, 0] = 9, 8, 12
    g[1, 1, 0] = 9, 8, 2
    g[0, 1, 1] = 3, 4, 6
    g[1, 0, 1] = 3, 4, 4
    game_dicts.append({'g': g})

    for d in game_dicts:
        d['payoff_max'] = max(
            player.payoff_array.max() for player in d['g'].players
        )

    return game_dicts


def _unit_mass_ray(g):
    ray = np.zeros(sum(g.nums_actions))
    ray[np.cumsum(g.nums_actions) - 1] = 1
    return ray


def _is_mixed_action_profile(x, nums_actions, tol=1e-12):
    if len(x) != len(nums_actions):
        return False
    for x_i, n in zip(x, nums_actions):
        if len(x_i) != n or (x_i < -tol).any() or \
           not np.isclose(x_i.sum(), 1, rtol=0, atol=tol):
            return False
    return True


class TestIPASolve:
    def setup_method(self):
        self.game_dicts = _gametracer_game_dicts()

    def test_ipa_solve(self):
        fuzz_default = 1e-6
        rng = np.random.default_rng(1234)
        for d in self.game_dicts:
            NE = ipa_solve(d['g'], rng=rng)
            # Heuristic; no epsilon-optimality guarantee derived in the
            # paper
            tol = fuzz_default * d['payoff_max']
            assert_(d['g'].is_nash(NE, tol=tol))

    def test_fuzz_option(self):
        fuzz = 1e-8
        rng = np.random.default_rng(1234)
        for d in self.game_dicts:
            NE = ipa_solve(d['g'], rng=rng, fuzz=fuzz)
            tol = fuzz * d['payoff_max']
            assert_(d['g'].is_nash(NE, tol=tol))

    def test_ray(self):
        g = self.game_dicts[1]['g']
        ray = _unit_mass_ray(g)
        NE, res = ipa_solve(g, ray=ray, full_output=True)
        assert_(np.array_equal(res.ray, ray))
        assert_(g.is_nash(NE, tol=1e-5))

    def test_rng_reproducibility(self):
        g = self.game_dicts[1]['g']
        NE0, res0 = ipa_solve(g, rng=1234, full_output=True)
        NE1, res1 = ipa_solve(g, rng=1234, full_output=True)
        assert_(np.array_equal(res0.ray, res1.ray))
        assert_(compare_act_profs(np.array_equal, NE0, NE1))

    def test_full_output(self):
        g = self.game_dicts[0]['g']
        NE, res = ipa_solve(g, rng=1234, full_output=True)
        assert_(compare_act_profs(np.array_equal, NE, res.NE))
        assert_(res.converged)
        assert_(res.ret_code == 1)
        assert_(res.num_iter >= 1)
        assert_(res.max_iter == 100000)
        assert_(res.ray.shape == (sum(g.nums_actions),))

    def test_max_iter(self):
        # 2x2x2 game: needs more than 10 iterations with this ray
        g = self.game_dicts[1]['g']
        ray = [0.3, 0.7, 0.6, 0.4, 0.2, 0.8]
        _, res = ipa_solve(g, ray=ray, full_output=True)
        assert_(res.converged)
        needed = res.num_iter
        assert_(needed > 10)

        # Iteration limit reached: the last iterate is returned
        for max_iter in [1, 10]:
            NE, res = ipa_solve(g, ray=ray, max_iter=max_iter,
                                full_output=True)
            assert_(not res.converged)
            assert_(res.ret_code == 0)
            assert_(res.num_iter == max_iter)
            assert_(res.max_iter == max_iter)
            assert_(_is_mixed_action_profile(NE, g.nums_actions))

        # Exactly enough iterations
        NE, res = ipa_solve(g, ray=ray, max_iter=needed, full_output=True)
        assert_(res.converged)
        assert_(res.num_iter == needed)
        assert_(g.is_nash(NE, tol=1e-5))

    def test_max_pivots(self):
        # 3x2 game: the Lemke-Howson path has exactly 5 pivots
        g = self.game_dicts[0]['g']
        ray = [0., 0., 1., 0., 1.]
        zh_init = np.array([1/3, 1/3, 1/3, 1/2, 1/2])
        NE, res = ipa_solve(g, ray=ray, zh_init=zh_init, max_pivots=4,
                            full_output=True)
        assert_(not res.converged)
        assert_(res.ret_code == 0)
        assert_(res.num_iter == 1)
        assert_(_is_mixed_action_profile(NE, g.nums_actions))

        NE, res = ipa_solve(g, ray=ray, zh_init=zh_init, max_pivots=5,
                            full_output=True)
        assert_(res.converged)
        assert_(res.num_iter == 1)
        assert_(g.is_nash(NE, tol=1e-6))

    def test_invalid_ray_length(self):
        g = self.game_dicts[0]['g']
        assert_raises(ValueError, ipa_solve, g,
                      ray=np.zeros(sum(g.nums_actions) - 1))

    def test_keyword_only(self):
        g = self.game_dicts[0]['g']
        assert_raises(TypeError, ipa_solve, g,
                      np.ones(sum(g.nums_actions)))

    def test_ray_immutable(self):
        g = self.game_dicts[0]['g']
        ray = np.ones(sum(g.nums_actions))
        _, res = ipa_solve(g, ray=ray, full_output=True)
        ray_used = res.ray.copy()
        ray[:] = 0
        assert_(np.array_equal(res.ray, ray_used))


class TestGNMSolve:
    def setup_method(self):
        self.game_dicts = _gametracer_game_dicts()

    def test_gnm_solve(self):
        rng = np.random.default_rng(1234)
        for d in self.game_dicts:
            NEs = gnm_solve(d['g'], rng=rng)
            assert_(len(NEs) >= 1)
            for NE in NEs:
                assert_(d['g'].is_nash(NE, tol=1e-5))

    def test_ray(self):
        g = self.game_dicts[1]['g']
        ray = _unit_mass_ray(g)
        NEs, res = gnm_solve(g, ray=ray, full_output=True)
        assert_(np.array_equal(res.ray, ray))
        for NE in NEs:
            assert_(g.is_nash(NE, tol=1e-5))

    def test_rng_reproducibility(self):
        g = self.game_dicts[1]['g']
        NEs0 = gnm_solve(g, rng=1234)
        NEs1 = gnm_solve(g, rng=1234)
        assert_(compare_lists_act_profs(np.array_equal, NEs0, NEs1))

    def test_full_output(self):
        g = self.game_dicts[0]['g']
        NEs, res = gnm_solve(g, rng=1234, full_output=True)
        assert_(res.ret_code == len(NEs))
        assert_(res.num_iter >= 1)
        assert_(res.max_iter == 5000)
        assert_(res.ray.shape == (sum(g.nums_actions),))

    def test_max_iter(self):
        # 3x2 game: the three equilibria are found one by one along the
        # path
        g = self.game_dicts[0]['g']
        ray = [0., 0., 1., 0., 1.]
        NEs, res = gnm_solve(g, ray=ray, full_output=True)
        assert_(len(NEs) == 3)
        needed = res.num_iter
        assert_(needed > 3)

        prev = 0
        for max_iter in range(1, needed):
            NEs, res = gnm_solve(g, ray=ray, max_iter=max_iter,
                                 full_output=True)
            assert_(0 <= len(NEs) <= 3)
            assert_(res.ret_code == len(NEs))
            assert_(len(NEs) >= prev)  # Found in path order
            assert_(res.num_iter == max_iter)
            assert_(res.max_iter == max_iter)
            for NE in NEs:
                assert_(g.is_nash(NE, tol=1e-8))
            prev = len(NEs)
        assert_(0 < prev < 3)  # The last crossing is needed for the third one

        NEs, res = gnm_solve(g, ray=ray, max_iter=needed, full_output=True)
        assert_(len(NEs) == 3)
        assert_(res.num_iter == needed)

    def test_invalid_ray_length(self):
        g = self.game_dicts[0]['g']
        assert_raises(ValueError, gnm_solve, g,
                      ray=np.zeros(sum(g.nums_actions) - 1))

    def test_keyword_only(self):
        g = self.game_dicts[0]['g']
        assert_raises(TypeError, gnm_solve, g,
                      np.ones(sum(g.nums_actions)))

    def test_ray_immutable(self):
        g = self.game_dicts[0]['g']
        ray = np.ones(sum(g.nums_actions))
        _, res = gnm_solve(g, ray=ray, full_output=True)
        ray_used = res.ray.copy()
        ray[:] = 0
        assert_(np.array_equal(res.ray, ray_used))


def _expected_payoffs(g, action_profile):
    N = g.N
    expected_payoffs = np.empty(N)
    for i, player in enumerate(g.players):
        if N == 2:
            opponents_actions = action_profile[1-i]
        else:
            opponents_actions = \
                tuple(action_profile[(i + j) % N] for j in range(1, N))
        expected_payoffs[i] = \
            action_profile[i] @ player.payoff_vector(opponents_actions)
    return expected_payoffs


class TestNashCP:
    def setup_method(self):
        self.game_dicts = _gametracer_game_dicts()

    def test_nash_cp(self):
        for d in self.game_dicts:
            NE = nash_cp(d['g'])
            assert_(d['g'].is_nash(NE, tol=1e-5))

    def test_init(self):
        g = self.game_dicts[1]['g']
        inits = [([1, 0], [1, 0], [1, 0]),
                 ([0, 1], [0, 1], [1, 0]),
                 ([0, 1], [1/3, 2/3], [1/3, 2/3])]
        NEs = []
        for init in inits:
            NE = nash_cp(g, init=init)
            assert_(g.is_nash(NE, tol=1e-5))
            NEs.append(NE)
        # Different initial points lead to different equilibria
        assert_(not compare_act_profs(np.allclose, NEs[0], NEs[1],
                                      atol=1e-5))

    def test_full_output(self):
        for d in self.game_dicts:
            g = d['g']
            NE, res = nash_cp(g, full_output=True)
            assert_(compare_act_profs(np.array_equal, NE, res.NE))
            assert_(res.status == 'LOCALLY_SOLVED')
            assert_(np.allclose(res.expected_payoffs,
                                _expected_payoffs(g, NE), atol=1e-5))

    def test_no_solution(self):
        # Matching pennies has no pure-action equilibrium
        g = NormalFormGame([[(1, -1), (-1, 1)], [(-1, 1), (1, -1)]])
        NE, res = nash_cp(g, allow_mixed=False, full_output=True)
        assert_(NE is None)
        assert_(res.NE is None)
        assert_(res.status == 'LOCALLY_INFEASIBLE')
        assert_(res.expected_payoffs.shape == (0,))

        NE = nash_cp(g)
        assert_(compare_act_profs(np.allclose, NE, ([1/2, 1/2], [1/2, 1/2]),
                                  atol=1e-5))

    def test_options(self):
        g = self.game_dicts[0]['g']
        for options in [dict(ϵ=1e-8), dict(verbosity='NONE'),
                        dict(domrem_strict=False, domrem_mixedactions=False,
                             domrem_tol=1e-8, domrem_solver='GLPK')]:
            NE = nash_cp(g, **options)
            assert_(g.is_nash(NE, tol=1e-5))

    def test_invalid_verbosity(self):
        g = self.game_dicts[0]['g']
        assert_raises(ValueError, nash_cp, g, verbosity='loud')

    def test_invalid_init(self):
        g = self.game_dicts[1]['g']
        assert_raises(ValueError, nash_cp, g, init=([1, 0], [1, 0]))
        assert_raises(ValueError, nash_cp, g,
                      init=([1, 0, 0], [1, 0], [1, 0]))

    def test_keyword_only(self):
        g = self.game_dicts[1]['g']
        assert_raises(TypeError, nash_cp, g, ([1, 0], [1, 0], [1, 0]))

    def test_init_immutable(self):
        g = self.game_dicts[1]['g']
        init = [np.array([1., 0.]) for i in range(3)]
        nash_cp(g, init=init)
        for x in init:
            assert_(np.array_equal(x, [1., 0.]))


class TestNashSE:
    def setup_method(self):
        self.game_dicts = _gametracer_game_dicts()
        # From von Stengel 2007 in Algorithmic Game Theory
        self.game_dicts[0]['NEs'] = [([1, 0, 0], [1, 0]),
                                     ([4/5, 1/5, 0], [2/3, 1/3]),
                                     ([0, 1/3, 2/3], [1/3, 2/3])]
        # 2x2x2 game from McKelvey and McLennan: 9 equilibria on 8
        # support profiles (two of them on the full support profile)
        self.game_dicts[1]['NEs'] = _three_player_game_dicts()[0]['NEs']

    def test_nash_se(self):
        for d in self.game_dicts:
            NEs = nash_se(d['g'])
            assert_(len(NEs) == 1)
            assert_(d['g'].is_nash(NEs[0], tol=1e-5))

    def test_all(self):
        d = self.game_dicts[0]
        NEs = nash_se(d['g'], max_samples=float('inf'))
        assert_(compare_lists_act_profs(
            np.allclose, NEs, d['NEs'], atol=1e-8
        )
        )

        d = self.game_dicts[1]
        NEs, res = nash_se(d['g'], max_samples=float('inf'),
                           full_output=True)
        assert_(len(NEs) == 8)
        assert_(len(set(
            tuple(tuple(s) for s in supports) for supports in res.supports
        )) == 8)  # One equilibrium per support profile
        for NE in NEs:
            assert_(any(compare_act_profs(np.allclose, NE, NE_true,
                                          atol=1e-5)
                        for NE_true in d['NEs']))

    def test_max_samples(self):
        g = self.game_dicts[1]['g']
        for max_samples in [1, 2, 3]:
            NEs = nash_se(g, max_samples=max_samples)
            assert_(len(NEs) == max_samples)
            for NE in NEs:
                assert_(g.is_nash(NE, tol=1e-5))

    def test_full_output(self):
        for d in self.game_dicts:
            g = d['g']
            NEs, res = nash_se(g, max_samples=float('inf'), full_output=True)
            assert_(compare_lists_act_profs(np.array_equal, NEs, res.NEs))
            assert_(len(res.expected_payoffs) == len(NEs))
            assert_(len(res.supports) == len(NEs))
            for NE, expected_payoffs, supports in zip(NEs,
                                                      res.expected_payoffs,
                                                      res.supports):
                assert_(np.allclose(expected_payoffs,
                                    _expected_payoffs(g, NE), atol=1e-5))
                for x, support in zip(NE, supports):
                    assert_(np.array_equal(np.flatnonzero(x > 1e-5),
                                           support))

    def test_isolated_eq_only(self):
        # Degenerate game with 3 extreme equilibria
        bimatrix = [[(3, 3), (3, 3)],
                    [(2, 2), (5, 6)],
                    [(0, 3), (6, 1)]]
        g = NormalFormGame(bimatrix)
        NEs_true = [([1, 0, 0], [1, 0]),
                    ([1, 0, 0], [2/3, 1/3]),
                    ([0, 1/3, 2/3], [1/3, 2/3])]

        # Only support profiles of equal sizes are enumerated
        NEs, res = nash_se(g, max_samples=float('inf'), full_output=True)
        assert_(all(len(s[0]) == len(s[1]) for s in res.supports))
        for NE in NEs:
            assert_(any(compare_act_profs(np.allclose, NE, NE_true,
                                          atol=1e-8)
                        for NE_true in NEs_true))

        # All support profiles are enumerated; an equilibrium may be
        # found on more than one support profile
        NEs, res = nash_se(g, max_samples=float('inf'),
                           isolated_eq_only=False, full_output=True)
        assert_(any(len(s[0]) != len(s[1]) for s in res.supports))
        for NE in NEs:
            assert_(any(compare_act_profs(np.allclose, NE, NE_true,
                                          atol=1e-8)
                        for NE_true in NEs_true))
        for NE_true in NEs_true:
            assert_(any(compare_act_profs(np.allclose, NE, NE_true,
                                          atol=1e-8)
                        for NE in NEs))

    def test_allow_mixed(self):
        d = self.game_dicts[0]
        NEs = nash_se(d['g'], max_samples=float('inf'), allow_mixed=False)
        assert_(compare_lists_act_profs(
            np.allclose, NEs, d['NEs'][:1], atol=1e-8
        )
        )

    def test_mt(self):
        g = self.game_dicts[1]['g']
        NEs_mt = nash_se(g, max_samples=float('inf'), mt=True)
        NEs_st = nash_se(g, max_samples=float('inf'), mt=False)
        assert_(compare_lists_act_profs(np.allclose, NEs_mt, NEs_st,
                                        atol=1e-8))

    def test_options(self):
        g = self.game_dicts[0]['g']
        for options in [dict(verbosity='NONE'),
                        dict(domrem_strict=False, domrem_mixedactions=False,
                             domrem_tol=1e-8, domrem_solver='GLPK')]:
            NEs = nash_se(g, **options)
            assert_(len(NEs) == 1)
            assert_(g.is_nash(NEs[0], tol=1e-5))

    def test_keyword_only(self):
        g = self.game_dicts[0]['g']
        assert_raises(TypeError, nash_se, g, 2)


def test_invalid_1player_g():
    g = NormalFormGame([[1], [2], [3]])
    for func in [lrsnash, support_enumeration, hc_solve, ipa_solve,
                 gnm_solve, nash_cp, nash_se]:
        assert_raises(NotImplementedError, func, g)


def test_invalid_input():
    bimatrix = [[(3, 3), (3, 2)],
                [(2, 2), (5, 6)],
                [(0, 3), (6, 1)]]
    for func in [lrsnash, support_enumeration, hc_solve, ipa_solve,
                 gnm_solve, nash_cp, nash_se]:
        assert_raises(TypeError, func, bimatrix)
