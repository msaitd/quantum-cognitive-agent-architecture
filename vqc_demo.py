# -*- coding: utf-8 -*-
"""
Second proof-of-concept for the QCA (Level 2 / L5): a parameterised quantum
circuit (VQC) policy trained by the parameter-shift policy gradient (Eqs. 16-18)
on a toy contextual-bandit task. Pure-numpy statevector simulation.

Task: 2-qubit context s in {0,1,2,3}; optimal action a*(s) = b0 XOR b1
(a non-linearly-separable mapping). Reward r(s,a) = 1 if a == a*(s) else 0.
The agent reads action a from a measurement of qubit 0 (Born rule), so
pi_theta(a=0|s) = P(qubit0 = 0). We ascend the expected reward
J(theta) = mean_s sum_a pi_theta(a|s) r(s,a) using the exact parameter-shift gradient.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(3)
I2 = np.eye(2, dtype=complex)
def RY(a):
    c, s = np.cos(a/2), np.sin(a/2)
    return np.array([[c, -s], [s, c]], dtype=complex)
def on0(G): return np.kron(G, I2)   # qubit 0 = high bit (index = 2*b0 + b1)
def on1(G): return np.kron(I2, G)
# CNOT, control = qubit1, target = qubit0  -> b0' = b0 XOR b1
CX = np.zeros((4, 4), dtype=complex)
for i in range(4):
    b0, b1 = i // 2, i % 2
    CX[2*(b0 ^ b1) + b1, i] = 1.0

def run(theta, s):
    psi = np.zeros(4, dtype=complex); psi[s] = 1.0
    L = theta.shape[0] - 1
    for l in range(L):
        psi = on0(RY(theta[l, 0])) @ psi
        psi = on1(RY(theta[l, 1])) @ psi
        psi = CX @ psi
    psi = on0(RY(theta[L, 0])) @ psi
    psi = on1(RY(theta[L, 1])) @ psi
    return psi

def p_a0(theta, s):
    psi = run(theta, s)
    return float(abs(psi[0])**2 + abs(psi[1])**2)   # P(qubit0 = 0)

aopt = [0, 1, 1, 0]   # b0 XOR b1
def J(theta):
    tot = 0.0
    for s in range(4):
        p0 = p_a0(theta, s); pol = [p0, 1 - p0]
        tot += pol[aopt[s]]
    return tot / 4.0

def grad_parameter_shift(theta):
    g = np.zeros_like(theta); sh = np.pi / 2
    for i in range(theta.shape[0]):
        for j in range(2):
            tp = theta.copy(); tp[i, j] += sh
            tm = theta.copy(); tm[i, j] -= sh
            g[i, j] = 0.5 * (J(tp) - J(tm))   # exact gradient (Eq. 18)
    return g

L = 3
theta = rng.uniform(-np.pi, np.pi, size=(L + 1, 2))
lr, iters = 0.35, 140
hist = []
for it in range(iters):
    hist.append(J(theta))
    theta = theta + lr * grad_parameter_shift(theta)
hist.append(J(theta))
hist = np.array(hist)
nparams = theta.size
print("VQC policy: %d qubits, %d trainable parameters" % (2, nparams))
print("expected reward: init %.3f  ->  final %.3f  (optimal = 1.0, random = 0.5)" % (hist[0], hist[-1]))

# ---- figure ----
fig, ax = plt.subplots(figsize=(7.0, 4.0))
ax.plot(hist, color="#2E5496", lw=2.2, label="VQC policy (parameter-shift gradient)")
ax.axhline(1.0, ls="--", color="#2E8B57", lw=1.4, label="optimal policy")
ax.axhline(0.5, ls=":", color="#6B6B6B", lw=1.4, label="random policy")
ax.set_xlabel("training iteration"); ax.set_ylabel("expected reward  J(θ)")
ax.set_ylim(0.4, 1.03)
ax.set_title("A few-qubit VQC policy learns a non-separable (XOR) task at the QCA's L5", fontsize=10.5)
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.text(0.02, 0.06, "2 qubits, %d parameters; reward %.2f → %.2f" % (nparams, hist[0], hist[-1]),
        transform=ax.transAxes, fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.4", fc="#F2F4F8", ec="#BBBBBB"))
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
np.savetxt(
    OUTPUT_DIR / "Figure5_data.csv",
    np.column_stack((np.arange(len(hist)), hist)),
    delimiter=",",
    header="iteration,expected_reward",
    comments="",
)
plt.savefig(OUTPUT_DIR / "Figure5_VQC_policy.png", dpi=300, facecolor="white")
plt.savefig(OUTPUT_DIR / "Figure5_VQC_policy.pdf", facecolor="white")
print(f"saved {OUTPUT_DIR / 'Figure5_VQC_policy.png'}")
print(f"saved {OUTPUT_DIR / 'Figure5_data.csv'}")
