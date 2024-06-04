import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.constants as const

plt.rcParams['text.usetex'] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.figsize"] = [4, 3]

def coef_to_mass(c):
    return 0.5*const.hbar**2*1e21/const.e/c

def dmdc(c):
    return -0.5*const.hbar**2*1e21/const.e/(c**2)

N = {"k": "k [nm-1]", "V": "V [mV]"}

df = pd.read_csv(sys.argv[1])

k = df[N['k']]*np.pi/2
E = df[N['V']]
mask = k == k
k = k[mask]
k2 = k**2
E = E[mask]
cutoffs = np.sort(k)[:1:-1]
print(cutoffs)
estimates = np.empty(cutoffs.shape + (2,))
stds = np.empty_like(estimates)
for n, ub in enumerate(cutoffs):
    fit_mask = k <= ub
    model = sm.OLS(E[fit_mask], sm.add_constant(k2[fit_mask]))
    result = model.fit()
    coef = result.params[N["k"]]
    estimates[n] = result.params
    stds[n] = result.bse
    print(result.summary())


m_effs = coef_to_mass(estimates[:,1])
m_eff_stds = np.abs(dmdc(estimates[:,1])*stds[:,1])

onset_result = sm.WLS(
    estimates[:,0],
    sm.add_constant(cutoffs),
    1/(stds[:,0]**2)
).fit()
print(onset_result.summary())
onset = onset_result.params[0]
onset_bounds = onset_result.conf_int()[0]

num_const = 1e31
m_eff_result = sm.WLS(
    m_effs*num_const,
    sm.add_constant(cutoffs),
    1/((m_eff_stds*num_const)**2)
).fit()
print(m_eff_result.summary())
m_eff = m_eff_result.params[0]/num_const
m_eff_bounds = m_eff_result.conf_int()[0]/num_const

plt.figure()
plt.errorbar(cutoffs, estimates[:,0], yerr=stds[:,0], marker='x', linestyle="none", color='k')
plt.axline(xy1 = (0,onset), slope=onset_result.params[1])
plt.xlabel("cutoff $k_{||}$ [nm$^{-1}$]")
plt.ylabel("surface state onset energy [meV]")
plt.title("Onset estimate as a function of cutoff $k_{||}$")
plt.tight_layout()

plt.figure()

slope = m_eff_result.params[1]/num_const / const.m_e
m_eff_m0 = m_eff / const.m_e
plot_m_effs = m_effs / const.m_e
plot_m_stds = m_eff_stds / const.m_e
plt.errorbar(
    cutoffs,
    m_effs / const.m_e,
    yerr=m_eff_stds / const.m_e,
    marker='x',
    linestyle="none",
    color='k'
)
plt.plot([0,4], [m_eff_m0,4*slope+m_eff_m0])
plt.xlabel("cutoff $k_{||}$ [nm$^{-1}$]")
plt.ylabel(r"best fit m$_{\text{eff}}$ [$m_e$]")
plt.title("Effective mass estimate as a function of cutoff $k_{||}$")
plt.tight_layout()

print(f"effective mass {m_eff} kg")
print(f"effective mass CI {m_eff_bounds/const.m_e} kg")
print(f"effective mass std {(m_eff_bounds-m_eff)/const.m_e} kg")
print(f"effective mass {m_eff/const.m_e}m0")
print(f"band onset at {onset} meV")
print(f"band onset CI {onset_bounds} kg")

plt.figure(figsize=(5.8, 4.35))
plt.scatter(k, E,color='k', label="measurement", marker='+')
plt.plot(
    np.sort(k),
    (onset
     + coef_to_mass(m_eff) * np.sort(k2)),
    label=r"\begin{equation*} \lim_{k\to0} \text{best fit} \end{equation*}"
)

fermi_index = np.nonzero(cutoffs < 2.2)[0][0]
plt.plot(
    np.sort(k),
    (estimates[fermi_index,0]
     + coef_to_mass(m_effs[fermi_index]) * np.sort(k2)),
    label=rf"\begin{{equation*}} \text{{cutoff }} k_{{||}}={cutoffs[fermi_index]:.3f} \text{{ nm}}^{{-1}}\text{{ best fit}} \end{{equation*}}"
)

plt.xlabel(r"$k_{||}$ [nm$^{-1}$]")
plt.ylabel("E [meV]")
plt.title("Cu(111) surface state dispersion from STM")
plt.legend()
plt.tight_layout()
plt.show()
