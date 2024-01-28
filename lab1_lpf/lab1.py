#%%
import math as m
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

# Transfer Function: abs(vout / vin) := 1 / sqrt(1 + (omegaa * R * C)**2)

#constants: 
R = 1188.7
C = 32.7 * 1e-9

#%%

fsample = [100, 300, 972, 3003, 10000, 30000, 100000]
theor_fsample = np.linspace(min(fsample), max(fsample), num=200, endpoint=True)


#error is 0.5% of value +- 1 in last digit
err_r = round(0.005 * R + 1)
err_c = 0.005 * C + 1e-9;

r, c, omega = sp.symbols('r c omega')

err_omega = 0.001 * omega

tFuncSP = sp.sqrt(1 + (2 * m.pi * omega * r * c) ** 2) ** -1


tFuncSPErr = sp.Abs(sp.diff(tFuncSP, r)) * err_r + sp.Abs(sp.diff(tFuncSP, c)) * err_c + sp.Abs(sp.diff(tFuncSP, omega)) * 0.001 * omega

#%%

#transfer Function
tFunc = lambda f : 1 / m.sqrt(1 + (2 * m.pi * f * R * C) ** 2) 

#convert magnitude in range [0, 1) to dB
dB = lambda f : 20 * m.log10(f)

# Calculate upper and lower bounds of theoretical transfer function
upper_bound = [tFunc(f) + tFuncSPErr.subs({r: R, c: C, omega: 2 * m.pi * f}) for f in theor_fsample]
lower_bound = [tFunc(f) - tFuncSPErr.subs({r: R, c: C, omega: 2 * m.pi * f}) for f in theor_fsample]
theor_err = [upper_bound[i] - lower_bound[i] for i in range(len(upper_bound))]


VinData = [19.8, 20.0, 20.0, 19.6, 19.2, 19.2, 19.4]
err_VinData = [0.2] * 7

VoutData = [19.6, 19.6, 18.8, 15.2, 7.2, 2.8, 0.82]
err_VoutData = [0.2] * 7

#%%

vin, vout, vinErr, voutErr = sp.symbols('vin vout vinErr voutErr')

#symbolic formula for experimental error propagation
expData = vout / vin


#experimental error propagation
expErr = sp.Abs(sp.diff(expData, vin) * vinErr) + sp.Abs(sp.diff(expData, vout) * voutErr)



Vout_Vin_Data = [expData.subs({vin: VinData[i], vout: VoutData[i]}).evalf() for i in range(len(VinData))]
Vout_Vin_Upper = [dB(Vout_Vin_Data[i] + expErr.subs({vin: VinData[i], vout: VoutData[i], vinErr: err_VinData[i], voutErr: err_VoutData[i]}).evalf()) for i in range(len(VinData))]
Vout_Vin_Lower = [dB(Vout_Vin_Data[i] - expErr.subs({vin: VinData[i], vout: VoutData[i], vinErr: err_VinData[i], voutErr: err_VoutData[i]}).evalf()) for i in range(len(VinData))]
Vout_Vin_Err = [Vout_Vin_Upper[i] - Vout_Vin_Lower[i] for i in range(len(VinData))]

#%%

plt.figure(dpi=500)

theor_data = list(map(lambda f : dB(tFunc(f)), theor_fsample))
#exper_data = list(map(dB, Vout_Vin_Data))
plt.semilogx(theor_fsample, theor_data, '-', label="Theor. Data W. Error")
plt.semilogx(theor_fsample, list(map(dB, upper_bound)), '--r', label="Theor. Upper Bound")
plt.semilogx(theor_fsample, list(map(dB, lower_bound)), '--b', label="Theor. Lower Bound")
plt.axvline(1/(2 * m.pi * R * C), linestyle='--', color='r', label='Cutoff Frequency')
plt.errorbar(fsample, list(map(dB, Vout_Vin_Data)), yerr=Vout_Vin_Err, fmt='.', label="Exp. Data w/ Error")


plt.legend()
plt.yticks(np.linspace(0, -30, 11))
plt.title("Theoretical Mag. of Transfer Function - Eqn (4)")
plt.ylabel("dB")
plt.xlabel("Frequency [Hz]")
plt.legend()
plt.show()


#%%

#Phase shift

rads = [f for f in fsample]
shift_ms = [0.000, 0.010, 0.032, 0.034, 0.020, 0.0076, 0.00220]
err_shift_ms = [0.004] * 5 + [0.0004] * 2

periods = [-1/f for f in fsample]
f_rad = [2 * m.pi * f for f in fsample]

#phase shift in radians
pshift = [2 * m.pi * (shift_ms[i] / 1000) / periods[i] for i in range(len(periods))]
err_pshift = [abs(2 * m.pi * (err_shift_ms[i] / 1000) / periods[i]) for i in range(len(periods))]

#%%
#Theoretical phase shift
tpshift = -sp.atan(2 * m.pi * omega * r * c)
err_tpshift = sp.Abs(sp.diff(tpshift, r) * err_r) + sp.Abs(sp.diff(tpshift, c) * err_c)
frange = np.logspace(2, 6, 200)

plt.figure(dpi=500)
plt.semilogx(frange, [tpshift.subs({r: R, c: C, omega: f}) for f in frange], label="Theoretical p-shift")
plt.semilogx(frange, [tpshift.subs({r: R, c: C, omega: f}) + err_tpshift.subs({r: R, c: C, omega: f}) for f in frange], '--', label="Upper bound on p-shift")
plt.semilogx(frange, [tpshift.subs({r: R, c: C, omega: f}) - err_tpshift.subs({r: R, c: C, omega: f}) for f in frange], '--', label="Upper bound on p-shift")
plt.axvline(1/(2 * m.pi * R * C), linestyle='--', color='blue', label="Cutoff freq.")
plt.errorbar(rads, pshift, yerr=err_pshift, label="Experimental p-shift values", fmt='.')
plt.ylabel("Phase shift (rad)")
ticks = [0, -m.pi/12, -m.pi/6, -m.pi/4, -m.pi/3, -5*m.pi/12, -m.pi/2]
labels = ['0', '$-\pi/12$', '$-\pi/6$', '$-\pi/4$', '$-\pi/3$', '$-5\pi/12$', '$-\pi/2$']
plt.yticks(ticks, labels)
plt.xlabel("Frequency (Hz)")
plt.title("Phase shift of transfer function")
plt.legend()
plt.show()

