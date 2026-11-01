"""Check analytic decomposition against complete multinomial enumeration."""
import math
import pytest
from hephaestus.evaluation.paired_holdout import exact_mcnemar_p
from hephaestus.evaluation.pilot_power import (analytic_paired_gate_power,binomial_probabilities,
    exact_binomial_interval,all_eos_gate_sensitivity)


@pytest.mark.parametrize('wins,losses',[(.1,.02),(.3,.1),(.7,0.),(0.,0.)])
def test_analytic_power_equals_complete_small_multinomial_enumeration(wins,losses):
    n=20;expected=0.
    for w in range(n+1):
      for l in range(n-w+1):
        if w-l<2 or l>2 or w<=l or exact_mcnemar_p(w,l)>=.2:continue
        ties=n-w-l
        expected+=math.comb(n,w)*math.comb(n-w,l)*wins**w*losses**l*(1-wins-losses)**ties
    observed=analytic_paired_gate_power(n=n,win_probability=wins,loss_probability=losses,
        alpha=.2,observed_gain_floor=.1,maximum_loss_fraction=.1)
    assert observed['subset_gate_power']==pytest.approx(expected,abs=1e-12)


def test_minimum_observed_gain_boundary_has_about_half_power_even_at_large_n():
    observed=analytic_paired_gate_power(n=2048,win_probability=.05,loss_probability=0.)
    assert .45<observed['subset_gate_power']<.55 and observed['numerical_omitted_mass']<1e-11
    assert .12<all_eos_gate_sensitivity(2048)[2]['probability_all_eos']<.14


def test_exact_intervals_cover_boundary_and_seed_reliability_values():
    assert exact_binomial_interval(0,64)[1]==pytest.approx(1-.025**(1/64),abs=1e-12)
    assert exact_binomial_interval(3,3)[0]==pytest.approx(.025**(1/3),abs=1e-12)
    assert sum(binomial_probabilities(2048,.05))==pytest.approx(1.,abs=2e-11)
