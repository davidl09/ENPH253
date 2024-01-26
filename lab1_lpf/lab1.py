#%%
import math as m
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

# Transfer Function: abs(vout / vin) := 1 / sqrt(1 + (omega * R * C)**2)

#constants: 

R = 1200 + 50
C = 33 * 1e-9

#%%

fsample = [100, 300, 972, 3003, 10000, 30000, 100000]

fToRad = lambda f : 2 * m.pi * f

r, c, omega = sp.symbols('r c omega')
err_r = 0.5 / 100 * r;
err_c = 0.5 / 100 * c;

tFuncSP = 1 / sp.sqrt(1 + (2 * m.pi * omega * r * c) ** 2)

tFuncSPErr = sp.diff(tFuncSP, r) * err_r + sp.diff(tFuncSP, c) * err_c

tFuncSPLambda = sp.lambdify(r, c, omega, tFuncSP)

tFunc = lambda f : 1 / m.sqrt(1 + (fToRad(f) * R * C) ** 2) 

dB = lambda f : 20 * m.log10(f)

VinData = [19.8, 20.0, 20.0, 19.6, 19.2, 19.2, 19.4]
VoutData = [19.6, 19.6, 18.8, 15.2, 7.2, 3.0, 0.82]

Vout_Vin_Data = [VoutData[i] / VinData[i] for i in range(len(VinData))]

#%%

plt.semilogx(fsample, list(map(lambda f : dB(tFunc(f)), fsample)))
plt.semilogx(fsample, list(map(dB, Vout_Vin_Data)), 'r.')
plt.yticks(np.linspace(0, -30, 11))
plt.title("Theoretical Mag. of Transfer Function - Eqn (4)")
plt.ylabel("dB")
plt.xlabel("Frequency [Hz]")
plt.show()