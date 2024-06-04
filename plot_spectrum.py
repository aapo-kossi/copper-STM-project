import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.figsize"] = [6, 4.5]

df = pd.read_csv(
    sys.argv[1],
    skiprows=629,
    sep="\t",
    index_col="index",
    usecols = [0,1,5],
    names = ["index", "bias", "didv"],
    encoding="ANSI",
)
df = df.iloc[511:]
plt.plot(df["bias"], df["didv"], marker="", label="measurements")#, color="k")
plt.xlabel("bias [mV]")
plt.ylabel(r"\begin{equation*} \frac{dI}{dV} \text{ [S]} \end{equation*}")
plt.title("Density of states spectrum of Cu(111) surface")
plt.axvline(-440, color='k', label="surface state onset $\mathrm{E}=-440$ meV")
plt.legend()
plt.tight_layout()
plt.show()
