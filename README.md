# Quantum Cognitive Agent Architecture

Code accompanying the manuscript:

**Toward a Quantum Cognitive Agent Architecture: From Quantum-Like Cognition to Hybrid Quantum-Classical AI and Quantum-Native Cognitive Agents**

This repository contains the Python code for two small, fully reproducible computational demonstrations for a proposed Quantum Cognitive Agent (QCA) reference architecture. The manuscript integrates quantum-like cognition, hybrid quantum-classical AI, quantum reinforcement learning, and speculative quantum-native agent design within a single epistemically stratified framework.

## Repository description

Suggested GitHub repository description:

> Formal Quantum Cognitive Agent architecture with simulations for quantum-like decision, hybrid QML/RL, and adaptive autonomous behaviour.

Repository name:

```text
quantum-cognitive-agent-architecture
```

## Computational demonstrations

### 1. QQ-equality decision-layer test

`code/simulation_qq.py` tests the parameter-free QQ-equality residual

```text
q = [p_AB(YY) + p_AB(NN)] - [p_BA(YY) + p_BA(NN)]
```

for:

- a quantum question-order model using projective measurements on a Hilbert space; and
- a generic non-quantum order-effect model with independent order-dependent carry-over biases.

The expected result is that the quantum decision layer satisfies the QQ equality to machine precision, whereas the generic order-effect model produces a broad residual distribution.

### 2. VQC policy-learning demonstration

`code/vqc_demo.py` implements a two-qubit variational quantum-circuit policy trained by the parameter-shift gradient on a toy XOR contextual-bandit task.

The expected result is that the policy learns the optimal action rule, with expected reward rising from approximately `0.41` to `1.00`.

## Quick start

Clone the repository and install the minimal Python dependencies:

```bash
git clone https://github.com/msaitd/quantum-cognitive-agent-architecture.git
cd quantum-cognitive-agent-architecture
python -m pip install numpy matplotlib
```

Run the demonstrations:

```bash
python code/simulation_qq.py
python code/vqc_demo.py
```


## Reproducibility

The demonstrations use fixed random seeds:

- `simulation_qq.py`: `20260625`
- `vqc_demo.py`: `3`

The scripts use only NumPy and Matplotlib and do not require a quantum-computing framework or quantum hardware.

## Data availability

No external or pre-existing datasets are used. The numerical data underlying the demonstrations are generated synthetically by the scripts in `code/`. Generated CSV and figure files are written to a local `outputs/` directory and are not version-controlled in this code-only repository.

## Ethical statement

The computational demonstrations use no human participants, human data, human tissue, animals, or external personal data.

## Citation

If you use this repository, please cite the associated manuscript:

```text
Dündar, M. S. (manuscript in submission). Toward a Quantum Cognitive Agent Architecture:
From Quantum-Like Cognition to Hybrid Quantum-Classical AI and Quantum-Native
Cognitive Agents.
```
