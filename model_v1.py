import math
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pymc.math as pmath
import numpy as np
from pymc.pytensorf import collect_default_updates, get_mode
import pytensor

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
        design_return = design(design_return)
        factors.append(design_return)

    return factors


# Simulate perfect dilution experiment with binomial model for basis

rng = np.random.default_rng()
az.style.use("arviz-darkgrid")

# True parameter values
E = 1e10  # PFUS in neat sample


# counts = [[] for k in range(12)]
# for b in range(1000):
#     C = []
#     C.append(E)
C = []
C.append(E)
DF_cum = [1]
DF_step = [1]
for ind in range(1, 12):
    # count = rng.poisson(C[ind - 1] * 0.1)
    count = rng.binomial(C[ind - 1], 0.1)
    C.append(count)
    DF_cum.append(DF_cum[ind - 1] * 0.1)
    DF_step.append(0.1)
    # counts[ind].append(count)


# fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
# axes[0].hist(counts[8], 20)
# axes[1].hist(counts[9], len(set(counts[9])))
# plt.show()
print(test_data["plaque count"].loc[~(test_data["plaque count"] == "tntc")])
print(C)
C_obs = np.array([c for c in C if c < 400])
print(C_obs)
df_obs = DF_cum[len(DF_cum) - len(C_obs) :]
print(df_obs)
G_max = np.max(C_obs / df_obs)
C = np.array(C)
G = G_max * np.array(DF_cum)
tr = np.min(np.nonzero(C < 1e6))
print(DF_step)
print(G[tr])
L, U = 8, 10
titration_model = pm.Model()


def rec_bin_count(*args):
    dilution_factor, previous_count = args
    next_count = pm.Binomial.dist(n=previous_count, p=dilution_factor)
    return next_count, collect_default_updates(inputs=args, outputs=[next_count])


with titration_model:
    obs_counts = pm.MutableData(
        "obs_counts",
        C_obs,
    )
    dilution_factors = pm.MutableData("dilution_factors", DF_step)
    # dilution_factors_ran = pm.Beta()
    e = pm.Gamma("e", mu=(L + U) / 2, sigma=(U - L) / 2)
    a = pm.Uniform("a", lower=0, upper=1)
    log10_N0_P1 = pm.Gamma("log10_N0_P1", mu=e, sigma=a)
    N0 = pm.Deterministic("N0", 10**log10_N0_P1 - 1)
    N5 = pm.Poisson("N1", N0 * 0.1**5)

    counts, updates = pytensor.scan(
        rec_bin_count, sequences=[np.array(DF_step)], outputs_info=[N5]
    )
    C = pm.Deterministic("C", counts)

    # C10 = pm.Deterministic("C10", pmath.switch(N0 < 10 ^ 7, C10b, C10p))
    titration_model.debug(verbose=True)
    smp = pm.sample_prior_predictive(samples=100, model=titration_model)

print(smp.prior)
