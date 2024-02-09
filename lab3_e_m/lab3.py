import numpy as np

#%%
#constants
true_e_m = 1.758820e11
#field strength correction for distance from center
B_B0_r8   = 0.99928
B_B0_r10  = 0.99621

K         = 7.73e-4

Kr_8      = K * B_B0_r8

Kr_10     = K * B_B0_r10


#%%
#data

I_8_ccw = { #multiply each voltage by 0.99 to subtract 1%
    150.0 * 0.99: 1.355,
    170.3 * 0.99: 1.446,
    189.8 * 0.99: 1.52,
    209.8 * 0.99: 1.604,
    230.2 * 0.99: 1.674,
    250.0 * 0.99: 1.752
}

I_8_cc = {
    149.8 * 0.99: 1.248,
    170.1 * 0.99: 1.325,
    190.0 * 0.99: 1.400,
    210.1 * 0.99: 1.474,
    230.5 * 0.99: 1.546,
    250.3 * 0.99: 1.613
}

I_10_ccw = {
    150.5 * 0.99: 1.117,
    170.3 * 0.99: 1.177,
    189.9 * 0.99: 1.234,
    210.3 * 0.99: 1.301,
    229.9 * 0.99: 1.367,
    250.6 * 0.99: 1.422
}

I_10_cc = {
    150.1 * 0.99: 0.960,
    170.6 * 0.99: 1.021,
    189.5 * 0.99: 1.089,
    210.4 * 0.99: 1.165,
    229.7 * 0.99: 1.205,
    250.5 * 0.99: 1.260
}

I_8_ccw  = np.array(list(I_8_ccw.items()), dtype=np.float128())
I_8_cc   = np.array(list(I_8_cc.items()), dtype=np.float128())
I_10_ccw = np.array(list(I_10_ccw.items()), dtype=np.float128())
I_10_cc  = np.array(list(I_10_cc.items()), dtype=np.float128())

#%%
#defining lambdas

Bt = lambda Il, Is, Kc : 0.5 * Kc * (Il + Is)
Be = lambda Il, Is, Kc : 0.5 * Kc * (Il - Is)

e_m = lambda V, B, r : 2 * V / (B ** 2 * r ** 2)

#field strengths
Bt_8  = Bt(I_8_ccw[:, 1], I_8_cc[:, 1], Kr_8)
avg_V_8 = (I_8_cc[:, 0] + I_8_ccw[:, 0]) / 2


Bt_10 = Bt(I_10_ccw[:, 1], I_10_cc[:, 1], Kr_10)
avg_V_10 = (I_10_cc[:, 0] + I_10_ccw[:, 0]) / 2

e_m_8 = e_m(avg_V_8, Bt_8, 0.04)
e_m_10 = e_m(avg_V_10, Bt_10, 0.05)

e_m_vals = np.concatenate((e_m_8, e_m_10))

em_ratio = e_m_vals.mean()

#%%

print(f"Average e/m value: {em_ratio:e}")
print(f"Standard deviation: {np.std(e_m_vals):e}")
print(f"Experiment error: {round(((em_ratio / true_e_m) - 1) * 100, 2)}%")
print(f"Allowable Error: {np.std(e_m_vals) / em_ratio * 100}%")

#%%
#Earth's magnetic field
Be_8 = Be(I_8_ccw[:, 1], I_8_cc[:, 1], Kr_8)
Be_10 = Be(I_10_ccw[:, 1], I_10_cc[:, 1], Kr_10)
Be_vals = np.concatenate((Be_8, Be_10))
Be_vals.mean()

print(f"Earth's magnetic field (calculated): {round(float(Be_vals.mean() * 1e9), 1)}")
print(f"Earth's magnetic Field (known value): 53122.6 nT")

#%%
e_mass = 1.60217663e-19/em_ratio
e_mass_known = 9.1093837e-31
print(f"Mass of electron (calculated): {e_mass}kg")
print(f"Mass of electron (known): {e_mass_known}kg")
