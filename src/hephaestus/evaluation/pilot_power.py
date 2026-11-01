"""Analytic planning calculations for an explicitly IID paired outcome model.

These calculations use no examples, model generation or authorization. They do
not certify power for weighted, stratified or clustered sampling, or accept a
scientific design. Probabilities use floating arithmetic, not interval arithmetic.
"""
from functools import lru_cache
import math


def binomial_probabilities(n,p):
    """Compute a normalized PMF outward from its mode, avoiding overflow."""
    if type(n) is not int or n<0 or not math.isfinite(p) or not 0<=p<=1:
        raise ValueError('invalid binomial parameters')
    values=[0.]*(n+1)
    if p in (0.,1.):values[0 if p==0 else n]=1.;return values
    mode=min(n,int((n+1)*p))
    values[mode]=1.
    for k in range(mode,0,-1):values[k-1]=values[k]*k/(n-k+1)*(1-p)/p
    for k in range(mode,n):values[k+1]=values[k]*(n-k)/(k+1)*p/(1-p)
    normalizer=math.fsum(values)
    return [value/normalizer for value in values]


@lru_cache(maxsize=32768)
def _positive_exact_critical(discordants,alpha):
    distribution=binomial_probabilities(discordants,.5);tail=0.;critical=discordants+1
    for wins in range(discordants,discordants//2,-1):
        tail+=distribution[wins]
        if 2*tail<alpha:critical=wins
        else:break
    return critical


def analytic_paired_gate_power(*,n,win_probability,loss_probability,alpha=.05,
                              observed_gain_floor=.05,maximum_loss_fraction=.01):
    """Sum D~Bin(n,w+l), W|D~Bin(D,w/(w+l)), without Monte Carlo trials.

    Compute positive two-sided exact McNemar rejection AND observed-gain/loss
    gates. The result upper-bounds their conjunction with additional gates only
    under this IID model; it is not full stratified/clustered experiment power.
    The omitted marginal mass is reported; floating roundoff is not certified.
    """
    if (type(n) is not int or not 1<=n<=8192 or not 0<alpha<1
            or any(not math.isfinite(p) or not 0<=p<=1 for p in
                   (win_probability,loss_probability,alpha,observed_gain_floor,maximum_loss_fraction))
            or win_probability+loss_probability>1):raise ValueError('invalid paired planning inputs')
    q=win_probability+loss_probability
    marginal=binomial_probabilities(n,q);gain_count=math.ceil(observed_gain_floor*n-1e-12)
    maximum_losses=math.floor(maximum_loss_fraction*n+1e-12);power,omitted=0.,0.
    for discordants,mass in enumerate(marginal):
        if q==0:break
        if mass<1e-15:omitted+=mass;continue
        critical=max(_positive_exact_critical(discordants,alpha),math.ceil((discordants+gain_count)/2),
                     discordants-maximum_losses)
        if critical<=discordants:
            power+=mass*math.fsum(binomial_probabilities(discordants,win_probability/q)[critical:])
    return {'n':n,'win_probability':win_probability,'loss_probability':loss_probability,
            'subset_gate_power':min(1.,max(0.,power)),'numerical_omitted_mass':omitted,
            'joint_power_upper_bound':min(1.,max(0.,power+omitted)),
            'roundoff_certified':False,
            'assumptions':'IID paired multinomial cases; excludes additional family, format, bootstrap and EOS gates'}


def exact_binomial_interval(successes,n,*,alpha=.05):
    """Equal-tailed Clopper-Pearson interval by binomial-tail inversion.

    Seed-population interpretation additionally requires IID random seeds;
    three fixed predeclared seed values alone do not establish that assumption.
    """
    if type(n) is not int or type(successes) is not int or not 0<=successes<=n or n<1 or not math.isfinite(alpha) or not 0<alpha<1:
        raise ValueError('invalid binomial interval inputs')
    def bisect(function,target):
        low,high=0.,1.
        for _ in range(64):
            middle=(low+high)/2
            if function(middle)<target:low=middle
            else:high=middle
        return (low+high)/2
    lower=0. if successes==0 else bisect(lambda p:math.fsum(binomial_probabilities(n,p)[successes:]),alpha/2)
    upper=1. if successes==n else bisect(lambda p:math.fsum(binomial_probabilities(n,p)[successes+1:]),1-alpha/2)
    return [lower,upper]


def all_eos_gate_sensitivity(n,success_probabilities=(.99,.995,.999,.9995,1.)):
    if (type(n) is not int or n<1 or any(not math.isfinite(p) or not 0<=p<=1 for p in success_probabilities)):
        raise ValueError('invalid EOS planning inputs')
    return [{'cases':n,'assumed_per_case_eos_probability':p,'probability_all_eos':p**n,
             'assumption':'independent termination; sensitivity only'} for p in success_probabilities]
