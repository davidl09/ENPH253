#%%
#import libraries

import numpy as np
from sklearn.linear_model import LinearRegression
import math as m
import matplotlib.pyplot as plt
import sympy as sp

#%%
#data from experiment
#Part 1a) 
freq_ind_x = np.array([200, 490, 663, 800, 916, 1020, 1114, 1200])

omega_sq = (freq_ind_x * 2 * m.pi) ** 2

vrms_ind = np.array([2.61, 3.45, 4.08, 4.49, 4.81, 5.08, 5.26, 5.43])
err_vrms_ind = 0.005 * vrms_ind + 0.02

irms_ind = np.array([91.73, 85.72, 82.53, 78.69, 75.44, 72.50, 69.93, 67.26])
err_irms_ind = 0.0075 * irms_ind + 0.02

irms_ind /= 1000
err_irms_ind /= 1000

abs_z2 = (vrms_ind / irms_ind) ** 2

#known quantities

Rind = 24.5
err_Rind = 0.07
Lind = 10 * 1e-3
err_Lind = 0.05 * Lind


#%%
#error propagation

V, I, dV, dI = sp.symbols('V I dV dI')
R_sq = (V / I) ** 2
err_R_sq = sp.Abs(sp.diff(R_sq, V)) * dV + sp.Abs(sp.diff(R_sq, I)) * dI

err_R_data = np.array([err_R_sq.subs({V: vrms_ind[i], I: irms_ind[i], dV: err_vrms_ind[i], dI: err_irms_ind[i]}).evalf() for i in range(len(vrms_ind))], dtype=np.float64)

weights = 1/(err_R_data ** 2) #weighted fit

#%%
#Theoretical impedance wrt. frequency

omega, L = sp.symbols('omega L')

Z_sq_L = sp.Abs(sp.I * omega * L + Rind) ** 2

Z_sq_bounds = sp.Abs(sp.diff(Z_sq_L, L)) * err_Lind

#%%

dataFit = LinearRegression().fit(omega_sq.reshape(-1, 1), abs_z2, sample_weight=np.ones(8))

predictFromData = np.vectorize(lambda omg : dataFit.predict(np.array(omg).reshape(-1, 1)))

residuals = np.array(predictFromData(omega_sq) - abs_z2)
stddev = np.std(residuals)
mean = np.mean(residuals)
#%%
#plot regression results for part 1a)

plt.figure(dpi=200)
plt.errorbar(omega_sq, abs_z2, yerr=err_R_data, fmt='r.', label="Measured Data")
plt.plot(omega_sq, predictFromData(np.linspace(omega_sq.min(), omega_sq.max(), 8)), 'b--', label="Line of Best Fit")
plt.title("Measured Impedance of Inductor with changing Frequency")
plt.xlabel("\u03c9^2  [s^-2]")
plt.ylabel("|Z| ^ 2 [\u03a9^2]")
plt.xticks(omega_sq)
plt.legend()
plt.show()

#%%
#results
print("")
print(f"Exp. determined value of R_l: {m.sqrt(dataFit.intercept_)}\u03a9")
Rl_meas = m.sqrt(dataFit.intercept_)
print(f"Exp. determined value of L: {m.sqrt(dataFit.coef_)}H")
L_meas = m.sqrt(dataFit.coef_)

#%%
# Part 1b)
freq_cap_x = np.array([208, 220, 235, 255, 280, 310, 350, 420, 560, 1200])
w_sq = freq_cap_x * 2 * m.pi

vrms_cap = np.array([7.48, 7.48, 7.48, 7.47, 7.46, 7.45, 7.43, 7.40, 7.50, 7.38])
err_vrms_cap = 0.005 * vrms_cap + 0.02

irms_cap = np.array([4.528, 4.795, 5.124, 5.545, 6.090, 6.745, 7.582, 9.060, 12.222, 25.68])
err_irms_cap = 0.0075 * irms_cap + 0.02

irms_cap /= 1000
err_irms_cap /= 1000

#known quantities
Ccap = 0.464 * 1e-6
err_Ccap = 0.003 * 1e-6

data_cap_sq = (vrms_cap / irms_cap) ** 2
err_cap_sq = (err_vrms_cap / err_irms_cap) ** 2

#%%
#Theoretical Error
C = sp.symbols('C')
theor_Zabs_c = sp.Abs(1 / (C * omega * sp.I)) ** 2
err_theor_Zabs_c = sp.diff(theor_Zabs_c, C) * err_Ccap

#%%
#Linear fit

capDataFit = LinearRegression().fit((w_sq ** -2).reshape(-1, 1), data_cap_sq, sample_weight = err_cap_sq ** -2)

predictCapImp = lambda w_sq_2 : capDataFit.coef_ * w_sq_2 + capDataFit.intercept_
wrange = (np.linspace(200, 2000, 200) * 2 * m.pi) ** -2

#%%
#plotting

plt.figure(dpi=200)
plt.title("Capacitor impedance with changing freq.")
plt.xlabel("\u03c9^-2  [s^2]")
plt.ylabel("|Z|^2 [\u03a9^2]")
plt.errorbar(w_sq ** -2, data_cap_sq, yerr=err_cap_sq, fmt='r.', label="Measured Data w. uncertainty")
plt.plot(wrange, predictCapImp(wrange), label="Line of best fit")
plt.legend()


print(f"Exp. determined value of C: {1e6 * (1/m.sqrt(capDataFit.coef_))} uF")
C_meas = 1/m.sqrt(capDataFit.coef_)


#%%
#Part B
#data

Vin = np.array([5.30, 4.85, 4.40, 3.95, 3.50, 3.05, 2.60, 2.15, 1.70, 1.25])
err_Vin = 0.005 * Vin + 0.02

irms = np.array([62.13, 56.88, 51.66, 46.46, 41.24, 36.03, 30.76, 25.58, 20.31, 15.062]) / 1000
err_irms = 0.0075 * irms + 0.00002

vin_s = np.array([7.25, 6.60, 5.95, 5.30, 4.65, 4.00, 3.35, 2.70, 2.05, 1.40])
err_vin_s = 0.005 * vin_s + 0.02

irms_s = np.array([25.61, 25.31, 21.02, 18.785, 16.483, 14.175, 11.865, 9.585, 7.271, 5.005]) / 1000
err_irms_s = irms_s * 0.0075 + 0.00002

#%%
#Line of best fit

parrFit = LinearRegression().fit(Vin.reshape(-1, 1), irms)
seriesFit = LinearRegression().fit(vin_s.reshape(-1, 1), irms_s)


#%%
#Parallel: Z = 1 / (1 / Z1 + 1 / Z2) 
plt.figure(dpi=400)
plt.title("Voltage/Current relationship for RLC circuit @ f=1000Hz")
plt.errorbar(Vin, irms, err_irms, err_Vin, fmt='r.', label="Parallel combination")
plt.plot(Vin, parrFit.predict(Vin.reshape(-1, 1)), 'g--', label="Line of best fit")
plt.errorbar(vin_s, irms_s, err_irms_s, err_vin_s, fmt='b.', label="Series Combination")
plt.plot(vin_s, seriesFit.predict(vin_s.reshape(-1, 1)), 'g--')
plt.ylabel("Measured Current [A]")
plt.xlabel("Applied Voltage Vin [V] (f=1000Hz)")
plt.legend()
plt.show()


#%%
#Part C

w = 1000 * 2 * m.pi
#Parallel:
ZL = Rl_meas + 1j * w * L_meas
ZC = 1 / (1j * w * C_meas)

ZP = 1 / (1 / ZL + 1 / ZC)
magZP = abs(ZP)
ZS = ZL + ZC
magZS = abs(ZS)

print("Part 2:")
print(f"|Z| in Parallel, theoretical: {abs(1 / (1 / (Rind + Lind * 2 * m.pi * 1000j) + 1/(1/(1000j * Ccap))))}")
print(f"|Z| in Parallel, calculated from (1): {magZP}")
print(f"|Z| in Parallel, measured: {1 / parrFit.coef_[0]}")
print(f"|Z| in Series, theoretical: {abs(1/(1j * 1000 * 2 * m.pi * Ccap) + 1j * 2 * m.pi * 1000 * Lind + Rind)}")
print(f"|Z| in Series, calculated from (1): {magZS}")
print(f"|Z| in Series, measured: {1 / seriesFit.coef_[0]}")
#%%
#Part D/E

V = Vin[0]

ILexpt = np.abs(V / ZL)
ICexpt = np.abs(V / ZC)

V_s = vin_s[0]

VLexpt = V_s * abs(ZL / (ZL + ZC))
VCexpt = V_s - VLexpt

print(f"Measured Current IL, IC: {73.72 / 1000}, {21.55 / 1000}")
print(f"Expected Currents IL, IC: {ILexpt}, {ICexpt}")

print(f"Measured Voltages VL, VC: {1.7645}, {8.852}")
print(f"Expected Voltages VL, VC: {VLexpt}, {VCexpt}")