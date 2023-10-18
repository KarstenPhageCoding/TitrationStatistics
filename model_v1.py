import math
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pymc.math as pmath
import numpy as np

test_data = pd.read_excel(
    "/home/kskogsholm/TitrationStatistics/TitrationStatistics/test.xlsx",
    engine="openpyxl",
)

print(test_data)


def bradley_design(dilution_factor):
    if round(dilution_factor, 5) >= 1e-4:
        return dilution_factor * 100
    if round(dilution_factor, 5) <= 1e-5:
        return dilution_factor * 10


def QC_design(dilution_factor):
    if round(math.log10(dilution_factor), 5) % 2 == 0:
        # power of 10 is even
        return dilution_factor * 100
    else:
        return dilution_factor * 10


def dilution_series(plated_dilution_factor: float, design) -> float:
    factors = []
    design_return = plated_dilution_factor
    while round(design_return, 5) != 1:
        print(design_return)
        design_return = design(design_return)
        factors.append(design_return)

    return factors


print(dilution_series(1e-9, QC_design))


# Simulate perfect dilution experiment with binomial model for basis

rng = np.random.default_rng()
az.style.use("arviz-darkgrid")

# True parameter values
E = 1e10  # PFUS in neat sample


counts = [[] for k in range(12)]
for b in range(1000):
    C = []
    C.append(E)

    for ind in range(1, 12):
        count = rng.binomial(C[ind - 1], 0.1)

        C.append(count)

        counts[ind].append(count)

print(counts)


fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
axes[0].hist(counts[8], 20)
axes[1].hist(counts[9], len(set(counts[9])))
plt.show()

L, U = 1e9, 1e11
titration_model = pm.Model()
with titration_model:
    e = pm.Uniform("e", lower=8, upper=12)
    a = pm.Uniform("a", lower=0, upper=10)
    log10_N0_P1 = pm.Gamma("log10_N0_P1", mu=e, sigma=a)
    N0 = pm.Deterministic("N0", 10**log10_N0_P1)
    # C10 = pm.Binomial("C10", n=N0, p=(1e-10))
    # pm.model_to_graphviz(titration_model)
    titration_model.debug(verbose=True)
    smp = pm.sample_prior_predictive(samples=100, model=titration_model)

print(smp.prior)
