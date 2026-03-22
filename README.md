# DC Circuit Solver & Tutor

A DC circuit analysis tool that combines a Python **Modified Nodal Analysis (MNA)** solver with **Claude Opus 4.6** as a step-by-step tutor.

Claude selects the right analysis method, walks through the solution by hand, then calls the Python solver to compute exact numerical answers — all streamed in real time.

---

## How it works

```
Your circuit description
        │
        ▼
 Claude Opus 4.6
 (adaptive thinking)
  ├─ Identifies circuit topology
  ├─ Selects analysis method
  │   KVL/KCL · Nodal · Mesh
  │   Superposition · Thevenin/Norton
  ├─ Sets up equations manually
  │
  ├──► solve_circuit tool
  │         │
  │    solver.py (numpy MNA)
  │    ┌─────────────────┐
  │    │ Parse netlist   │
  │    │ Build G matrix  │
  │    │ Stamp R, V, I   │
  │    │ numpy.linalg    │
  │    │   .solve()      │
  │    └────────┬────────┘
  │             │ node voltages
  │             │ branch currents
  │             │ power dissipation
  │◄────────────┘
  │
  └─ Interprets results
     Explains physics
     Verifies power balance
```

---

## Project structure

```
dc_circuit_tutor/
├── solver.py        # MNA solver + Thevenin/Norton (pure Python + numpy)
├── tutor.py         # Claude API integration — streaming, tool use loop
├── main.py          # CLI entry point + unit tests
└── requirements.txt
```

### `solver.py` — the math

Implements **Modified Nodal Analysis** from scratch:

- Netlist parser with SI prefix support (`k`, `M`, `m`, `u`, `n`, `p`)
- MNA matrix construction using the stamp method for R, V, I elements
- Singularity / floating-node detection via condition number check
- Thevenin equivalent via open-circuit voltage + short-circuit current
- Norton equivalent derived automatically

### `tutor.py` — the Claude integration

- Claude Opus 4.6 with `thinking: {type: "adaptive"}`
- Streaming output (text appears as it is generated)
- Agentic tool-use loop: Claude calls `solve_circuit` and `compute_thevenin` tools backed by the Python solver
- Results fed back to Claude for physical interpretation

---

## Setup

```bash
cd dc_circuit_tutor
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
```

**Requirements:** Python 3.10+, `anthropic>=0.40.0`, `numpy>=1.24.0`

---

## Usage

### Run a built-in example

```bash
python main.py --example voltage_divider
python main.py --example wheatstone
python main.py --example thevenin
python main.py --example superposition
```

### Describe your own circuit

```bash
python main.py "Find the current through each resistor: 9V source, R1=3kΩ and R2=6kΩ in series"
```

### Interactive mode

```bash
python main.py
# Enter circuit description, press Enter twice to submit
```

### Verify the solver (no API key needed)

```bash
python main.py --test
```

---

## Netlist format

The tools accept a simple SPICE-like netlist. One component per line:

```
<name> <node+> <node-> <value>   # optional comment
```

| Prefix | Type | Example | Meaning |
|--------|------|---------|---------|
| `R` | Resistor | `R1 1 2 4k` | 4 kΩ between nodes 1 and 2 |
| `V` | Voltage source | `V1 1 0 12` | 12 V, node 1 positive, node 0 ground |
| `I` | Current source | `I1 0 2 2m` | 2 mA flowing from node 0 into node 2 |

- Ground is always node `0`
- Node labels can be numbers or names (`A`, `out`, `Vcc`)
- SI prefixes: `G` `M` `k` `m` `u` `n` `p`

**Voltage divider example:**
```
V1 1 0 12      # 12 V source
R1 1 2 4k      # 4 kΩ top resistor
R2 2 0 6k      # 6 kΩ bottom resistor
```

---

## Built-in examples

| Example | Circuit | Concepts covered |
|---------|---------|-----------------|
| `voltage_divider` | 12 V source, R1=4 kΩ, R2=6 kΩ | KVL, current, power balance |
| `wheatstone` | 5-resistor bridge, 10 V source | Nodal analysis, bridge balance condition |
| `thevenin` | 3-resistor network, 15 V source | Thevenin/Norton equivalent, load analysis |
| `superposition` | V source + I source, 2 resistors | Superposition, verification with nodal |

---

## MNA solver — technical notes

The solver implements the standard MNA stamp method. For a circuit with *n* non-ground nodes and *m* voltage sources, it builds and solves:

```
[G  B] [v]   [I_s]
[C  D] [j] = [V_s]
```

Where `G` is the conductance sub-matrix, `B`/`C` couple voltage sources into the node equations, `v` is the unknown node voltage vector, and `j` is the vector of unknown currents through voltage sources.

**Current source convention (SPICE):** `I n+ n- VALUE` — conventional current flows from `n+` to `n-` through the external circuit (i.e., into node `n-`).

**Thevenin algorithm:**
1. Solve original circuit → V_th = V(terminal+) − V(terminal−)
2. Insert 0 V ammeter across terminals → solve for I_sc
3. R_th = V_th / I_sc
