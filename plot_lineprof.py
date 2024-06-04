import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.figsize"] = [6, 4.5]

df = pd.read_csv(sys.argv[1],sep=";",header=[0,1])
df = df[df.columns[:-1]]
df = df.iloc[1:].astype(float) * 1e9

for x, y in zip(df.columns[:-1:2], df.columns[1::2]):
    plt.plot(df[x], df[y], label = x[0])
plt.xlabel("x [nm]")
plt.ylabel("y [nm]")
plt.title("Tip height line profiles")
plt.legend()
plt.tight_layout()
plt.show()
