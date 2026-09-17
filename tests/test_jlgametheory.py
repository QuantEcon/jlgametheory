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


def _is_mixed_action_profile(x, nums_actions, tol=1e-12):
    if len(x) != len(nums_actions):
        return False
    for x_i, n in zip(x, nums_actions):
        if len(x_i) != n or (x_i < -tol).any() or \
           not np.isclose(x_i.sum(), 1, atol=tol):
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
            assert_(len(NEs) >= prev)  # Found in path order
            assert_(res.num_iter == max_iter)
            assert_(res.max_iter == max_iter)
            for NE in NEs:
                assert_(g.is_nash(NE, tol=1e-8))
            prev = len(NEs)
        assert_(prev < 3)  # The last crossing is needed for the third one

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
