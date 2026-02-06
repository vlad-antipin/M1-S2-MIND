Exploratory vs theoretical vs empirical research

Comparison of central performances (mean / median)
Reject / not reject H0 
p-value = proba of wrongly rejecting H0 (FP)

Tests and assumptions:
Student's T - equal 
Non-parametric like WMW rank sum test
Bootstrap, permutation tests - expensive
Welch's T-test - the best one

slide 13 - pointwise comparisons between performances during training, but need multiple comparison correction

15 seeds seems to be enough, run 10 experiments by seed????? wtf? seed must make everything deterministic non??? #question 

Performance profiles #todo 

`rliable` module

IQM - Interquartile Mean - mean of central values - more reliable than just mean, robust to outliers

# Tuning methods

## Bayesian optimization

Start with a prior: performance is independent of parameter ( and equal to e.g. mean) with some variability slide 24

Take a parameter, measure performance, update the distribution - for the next point take point with highest uncertainty (upper confidence bound)

Method is sequential since choice of new point depends on previous choices - we choose the point that minimizes the uncertainty the most

NB: what chenges to account for noise? slide 27

Grid search - not sequential, thus can be sequential
Random search is usually better.......... how come? #lookup 

**Evolutionary methods**: explore better "sweet spots" - best of sequential and parallel - guide towards zones that work better - used the most

`optuna` #lookup  for those methods

need to automatize it, although for learning in uni play around yourself

contextualize performance: random (dummy), oracle (human or state of the art?)

Sensitivity curves: evolution of performance as parameter changes, analyse the shape 

In RL, **put steps and not episodes** on x-axis! since agents learn at each step, and episode durations can variate!!!

Dopaminergic neurons seems to follow TD error, slide 46





