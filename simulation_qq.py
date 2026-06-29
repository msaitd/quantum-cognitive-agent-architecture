# -*- coding: utf-8 -*-
"""
Illustrative proof-of-concept for the QCA decision layer (Eq. 15).

Numerically tests the parameter-free QQ-equality
    q = [p_AB(YY) + p_AB(NN)] - [p_BA(YY) + p_BA(NN)]
for (i) the quantum question-order model (projective measurements on a Hilbert
space) and (ii) a generic non-quantum order-effect model with independent,
order-dependent carry-over biases.

Result: the quantum decision layer satisfies q = 0 to machine precision for
arbitrary states and questions, whereas the generic order model does not.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(20260625)

def random_state(n):
    v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return v / np.linalg.norm(v)

def random_projector(n, k):
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)
    V = Q[:, :k]
    return V @ V.conj().T

def qq_quantum(n, trials):
    qs = np.empty(trials)
    I = np.eye(n)
    for t in range(trials):
        psi = random_state(n)
        Pa = random_projector(n, int(rng.integers(1, n)))   # 'yes' projector for A
        Pb = random_projector(n, int(rng.integers(1, n)))   # 'yes' projector for B
        Na, Nb = I - Pa, I - Pb                              # 'no' projectors
        # order A->B (Born + Lueders): p(answer1, answer2)
        pAB_YY = np.linalg.norm(Pb @ Pa @ psi) ** 2
        pAB_NN = np.linalg.norm(Nb @ Na @ psi) ** 2
        # order B->A
        pBA_YY = np.linalg.norm(Pa @ Pb @ psi) ** 2
        pBA_NN = np.linalg.norm(Na @ Nb @ psi) ** 2
        qs[t] = (pAB_YY + pAB_NN) - (pBA_YY + pBA_NN)
    return qs

def qq_classical(trials, bias=0.20):
    """Generic order-effect model: agreement probabilities are base joint rates plus an independent, order-dependent carry-over bias (anchoring/assimilation)."""
    qs = np.empty(trials)
    for t in range(trials):
        a = rng.uniform(0, 1); b = rng.uniform(0, 1)
        base_YY = a * b
        base_NN = (1 - a) * (1 - b)
        e1 = rng.uniform(-bias, bias)   # carry-over when A asked first
        e2 = rng.uniform(-bias, bias)   # carry-over when B asked first
        pAB_YY = np.clip(base_YY + e1, 0, 1); pAB_NN = np.clip(base_NN + e1, 0, 1)
        pBA_YY = np.clip(base_YY + e2, 0, 1); pBA_NN = np.clip(base_NN + e2, 0, 1)
        qs[t] = (pAB_YY + pAB_NN) - (pBA_YY + pBA_NN)
    return qs

N = 5000
q_q = qq_quantum(n=4, trials=N)
q_c = qq_classical(trials=N)

print("=== QQ-equality test (N=%d random instances) ===" % N)
print("Quantum model:   max|q| = %.3e   mean|q| = %.3e" % (np.max(np.abs(q_q)), np.mean(np.abs(q_q))))
print("Classical model: mean|q| = %.3f    std(q) = %.3f    P(|q|>0.05) = %.1f%%"
      % (np.mean(np.abs(q_c)), np.std(q_c), 100 * np.mean(np.abs(q_c) > 0.05)))

# ---- figure ----
fig, ax = plt.subplots(figsize=(7.2, 4.3))
ax.hist(q_c, bins=60, color="#C9821E", alpha=0.75, edgecolor="white", label="Generic order-effect model")
ax.axvline(0, color="#2E8B57", lw=3, label="Quantum decision layer  (q = 0, parameter-free)")
ax.set_xlabel("QQ-equality residual  q = [p$_{AB}$(YY)+p$_{AB}$(NN)] − [p$_{BA}$(YY)+p$_{BA}$(NN)]")
ax.set_ylabel("count (of %d random instances)" % N)
ax.set_title("The QQ-equality as an emergent, parameter-free signature of the QCA decision layer", fontsize=10.5)
ax.legend(frameon=False, fontsize=9, loc="upper left")
txt = "Quantum model: max|q| = %.1e  (machine precision)\nGeneric model: std(q) = %.2f,  P(|q|>0.05) = %.0f%%" % (
    np.max(np.abs(q_q)), np.std(q_c), 100 * np.mean(np.abs(q_c) > 0.05))
ax.text(0.985, 0.97, txt, transform=ax.transAxes, va="top", ha="right", fontsize=8.4,
        bbox=dict(boxstyle="round,pad=0.4", fc="#F2F4F8", ec="#BBBBBB"))
plt.tight_layout()
np.savetxt(
    OUTPUT_DIR / "Figure4_data.csv",
    np.column_stack((q_q, q_c)),
    delimiter=",",
    header="quantum_residual,generic_order_effect_residual",
    comments="",
)
plt.savefig(OUTPUT_DIR / "Figure4_QQ_equality.png", dpi=300, facecolor="white")
plt.savefig(OUTPUT_DIR / "Figure4_QQ_equality.pdf", facecolor="white")
print(f"saved {OUTPUT_DIR / 'Figure4_QQ_equality.png'}")
print(f"saved {OUTPUT_DIR / 'Figure4_data.csv'}")
