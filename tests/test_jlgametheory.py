from fractions import Fraction
import numpy as np
from numpy.testing import assert_, assert_raises
from quantecon.game_theory import NormalFormGame
from jlgametheory import lrsnash, hc_solve


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


def test_invalid_1player_g():
    g = NormalFormGame([[1], [2], [3]])
    for func in [lrsnash, hc_solve]:
        assert_raises(NotImplementedError, func, g)


def test_invalid_input():
    bimatrix = [[(3, 3), (3, 2)],
                [(2, 2), (5, 6)],
                [(0, 3), (6, 1)]]
    for func in [lrsnash, hc_solve]:
        assert_raises(TypeError, func, bimatrix)
