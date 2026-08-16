from fractions import Fraction
import numpy as np
from numpy.testing import assert_, assert_raises
from quantecon.game_theory import NormalFormGame
from jlgametheory import lrsnash, hc_solve, ipa_solve, gnm_solve


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
    for prof2 in list_act_profs2:
        if not any(compare_act_profs(operator, prof1, prof2, *args, **kwargs)
                   for prof1 in list_act_profs1):
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


class TestHCSolve:
    def setup_method(self):
        self.game_dicts = []

        # From von Stengel 2007 in Algorithmic Game Theory
        bimatrix = [[(3, 3), (3, 2)],
                    [(2, 2), (5, 6)],
                    [(0, 3), (6, 1)]]
        d = {'g': NormalFormGame(bimatrix),
             'NEs': [([1, 0, 0], [1, 0]),
                     ([4/5, 1/5, 0], [2/3, 1/3]),
                     ([0, 1/3, 2/3], [1/3, 2/3])]}
        self.game_dicts.append(d)

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
        d = {'g': g,
             'NEs': NEs}
        self.game_dicts.append(d)

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
        d = {'g': NormalFormGame(payoff_profile_array),
             'NEs': [([p, 1-p], [q, 1-q], [r, 1-r])]}
        self.game_dicts.append(d)

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
        assert_(res.ret_code == 1)
        assert_(res.ray.shape == (sum(g.nums_actions),))

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
        assert_(res.ray.shape == (sum(g.nums_actions),))

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


def test_invalid_1player_g():
    g = NormalFormGame([[1], [2], [3]])
    for func in [lrsnash, hc_solve, ipa_solve, gnm_solve]:
        assert_raises(NotImplementedError, func, g)


def test_invalid_input():
    bimatrix = [[(3, 3), (3, 2)],
                [(2, 2), (5, 6)],
                [(0, 3), (6, 1)]]
    for func in [lrsnash, hc_solve, ipa_solve, gnm_solve]:
        assert_raises(TypeError, func, bimatrix)
