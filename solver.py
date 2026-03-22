"""
DC Circuit Solver — Modified Nodal Analysis (MNA)

Implements a general-purpose DC circuit solver using the MNA stamp method.
Supports resistors (R), independent voltage sources (V), and current sources (I).

The MNA equation:  [G  B] [v]   [I_s]
                   [C  D] [j] = [V_s]

Where:
  G = conductance matrix (n×n, n = non-ground nodes)
  B = voltage source incidence into nodes (n×m, m = voltage sources)
  C = B^T (for passive networks)
  D = 0 matrix
  v = unknown node voltages
  j = unknown currents through voltage sources
  I_s = independent current source contributions
  V_s = independent voltage source values
"""

import re
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ─── Component & Netlist Parsing ─────────────────────────────────────────────

_SI_PREFIXES = {
    "G": 1e9, "M": 1e6, "k": 1e3, "K": 1e3,
    "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12,
}


def parse_value(s: str) -> float:
    """Parse a component value string with optional SI prefix.

    Examples:
        "10k"  → 10000.0
        "4.7u" → 0.0000047
        "12"   → 12.0
    """
    s = s.strip()
    for prefix, mult in _SI_PREFIXES.items():
        if s.endswith(prefix):
            return float(s[:-1]) * mult
    return float(s)


@dataclass
class Component:
    """A single circuit element."""
    name: str        # e.g. "R1", "V_in", "I2"
    node_pos: str    # positive terminal node (string label)
    node_neg: str    # negative terminal node (string label)
    value: float     # resistance [Ω], voltage [V], or current [A]

    @property
    def kind(self) -> str:
        """First letter of name, uppercased: 'R', 'V', or 'I'."""
        return self.name[0].upper()


def parse_netlist(netlist: str) -> List[Component]:
    """Parse a SPICE-like netlist string into a list of Components.

    Format (one component per line):
        <name> <node+> <node-> <value>  [# optional comment]

    Examples:
        R1 1 2 1000      # 1 kΩ resistor between nodes 1 and 2
        V1 1 0 12        # 12 V source, node 1 positive, node 0 ground
        I1 0 3 0.002     # 2 mA current source flowing from 0 into node 3
        R_load 2 0 4.7k  # 4.7 kΩ load to ground
    """
    components: List[Component] = []
    for raw_line in netlist.strip().splitlines():
        line = raw_line.split("#")[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        name, node_pos, node_neg, val_str = parts[0], parts[1], parts[2], parts[3]
        try:
            value = parse_value(val_str)
        except ValueError:
            continue
        components.append(Component(name=name, node_pos=node_pos, node_neg=node_neg, value=value))
    return components


# ─── Modified Nodal Analysis (MNA) Solver ────────────────────────────────────

class MNASolver:
    """General DC circuit solver using the MNA stamp method."""

    GROUND = "0"

    def __init__(self, components: List[Component]):
        self.components = components
        self.resistors  = [c for c in components if c.kind == "R"]
        self.vsources   = [c for c in components if c.kind == "V"]
        self.isources   = [c for c in components if c.kind == "I"]

        # Collect all nodes; ground is excluded from the unknown vector
        all_nodes: set = set()
        for c in components:
            all_nodes.add(c.node_pos)
            all_nodes.add(c.node_neg)

        self.nodes: List[str] = sorted(all_nodes - {self.GROUND})
        self.node_idx: Dict[str, int] = {n: i for i, n in enumerate(self.nodes)}

        self.n = len(self.nodes)     # non-ground nodes
        self.m = len(self.vsources)  # voltage source unknowns
        self.size = self.n + self.m

    def _idx(self, node: str) -> Optional[int]:
        """Return MNA row/column index for a node, or None for ground."""
        return None if node == self.GROUND else self.node_idx[node]

    def _build_matrix(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Construct the MNA matrix A and right-hand-side vector z.

        Returns:
            A     : (size × size) float array
            z     : (size,) float array
            steps : list of human-readable stamping steps (for pedagogy)
        """
        steps: List[str] = []
        A = np.zeros((self.size, self.size), dtype=float)
        z = np.zeros(self.size, dtype=float)

        steps.append(
            f"Matrix size: {self.size}×{self.size}  "
            f"({self.n} node voltage unknown(s) + {self.m} voltage-source current unknown(s))"
        )

        # ── Resistor stamps (conductance G = 1/R) ─────────────────────────
        steps.append("\nResistor stamps (G = 1/R added to conductance sub-matrix):")
        for r in self.resistors:
            G  = 1.0 / r.value
            ai = self._idx(r.node_pos)
            bi = self._idx(r.node_neg)
            if ai is not None: A[ai, ai] += G
            if bi is not None: A[bi, bi] += G
            if ai is not None and bi is not None:
                A[ai, bi] -= G
                A[bi, ai] -= G
            steps.append(
                f"  {r.name} = {r.value:.6g} Ω → G = {G:.6g} S  "
                f"(nodes {r.node_pos} ↔ {r.node_neg})"
            )

        # ── Voltage source stamps (B/C sub-matrices) ──────────────────────
        steps.append("\nVoltage source stamps (B and C sub-matrices):")
        for k, v in enumerate(self.vsources):
            row = self.n + k
            ai  = self._idx(v.node_pos)
            bi  = self._idx(v.node_neg)
            if ai is not None:
                A[ai, row] =  1.0   # B block: KCL contribution
                A[row, ai] =  1.0   # C block: KVL constraint
            if bi is not None:
                A[bi, row] = -1.0
                A[row, bi] = -1.0
            z[row] = v.value
            steps.append(
                f"  {v.name} = {v.value:.6g} V  "
                f"(node {v.node_pos} [+] → node {v.node_neg} [-])"
            )

        # ── Current source stamps (RHS vector z) ──────────────────────────
        # SPICE convention:  I n+ n- VALUE  →  current flows from n+ to n-
        # through the EXTERNAL circuit (exits n+, enters n-).
        # MNA z-vector encodes current injections INTO each node:
        #   z[n+] -= VALUE   (current leaves n+)
        #   z[n-] += VALUE   (current enters n-)
        if self.isources:
            steps.append("\nCurrent source stamps (into right-hand-side vector):")
        for i in self.isources:
            ai = self._idx(i.node_pos)
            bi = self._idx(i.node_neg)
            if ai is not None: z[ai] -= i.value   # current exits n+
            if bi is not None: z[bi] += i.value   # current enters n-
            steps.append(
                f"  {i.name} = {i.value:.6g} A  "
                f"(flows from node {i.node_pos} through external circuit into node {i.node_neg})"
            )

        return A, z, steps

    def solve(self) -> dict:
        """Solve the circuit; returns a results dict or {'error': ...}."""
        A, z, build_steps = self._build_matrix()

        # ── Singular / ill-conditioned check ──────────────────────────────
        try:
            cond = np.linalg.cond(A)
            if cond > 1e12:
                return {
                    "error": (
                        f"Matrix appears singular (condition number ≈ {cond:.2e}). "
                        "Common causes: floating node (no DC path to ground), "
                        "voltage source forming a loop, or disconnected circuit."
                    )
                }
        except Exception:
            pass

        try:
            x = np.linalg.solve(A, z)
        except np.linalg.LinAlgError as exc:
            return {"error": f"Singular matrix: {exc}. Check circuit topology."}

        # ── Extract node voltages ──────────────────────────────────────────
        node_voltages: Dict[str, float] = {self.GROUND: 0.0}
        for node, idx in self.node_idx.items():
            node_voltages[node] = round(float(x[idx]), 9)

        # ── Extract voltage-source branch currents ─────────────────────────
        vsource_currents: Dict[str, float] = {}
        for k, v in enumerate(self.vsources):
            vsource_currents[v.name] = round(float(x[self.n + k]), 9)

        # ── Compute resistor branch currents and power ─────────────────────
        branch_currents: Dict[str, float] = dict(vsource_currents)
        power_resistors: Dict[str, float] = {}
        for r in self.resistors:
            vp = node_voltages[r.node_pos]
            vn = node_voltages[r.node_neg]
            i  = (vp - vn) / r.value
            branch_currents[r.name] = round(i, 9)
            power_resistors[r.name] = round(i ** 2 * r.value, 9)

        total_dissipated = sum(power_resistors.values())
        total_supplied   = round(sum(
            -v.value * vsource_currents[v.name] for v in self.vsources
        ), 9)

        return {
            "node_voltages":        node_voltages,
            "branch_currents":      branch_currents,
            "power_resistors_W":    power_resistors,
            "total_dissipated_W":   total_dissipated,
            "total_supplied_W":     total_supplied,
            "mna_build_steps":      build_steps,
            "method":               "Modified Nodal Analysis (MNA)",
            "component_summary": {
                "resistors":       len(self.resistors),
                "voltage_sources": len(self.vsources),
                "current_sources": len(self.isources),
                "total_nodes":     len(self.nodes) + 1,
                "mesh_count":      max(0, len(self.components) - len(self.nodes)),
            },
        }


# ─── Thevenin / Norton Equivalent ────────────────────────────────────────────

def compute_thevenin(netlist: str, terminal_pos: str, terminal_neg: str) -> dict:
    """Compute the Thevenin (and Norton) equivalent seen at two terminals.

    Algorithm:
      1. V_th  = open-circuit voltage  (solve original circuit)
      2. I_sc  = short-circuit current (add 0 V test source between terminals)
      3. R_th  = V_th / I_sc

    Args:
        netlist:      SPICE-like netlist string (ground = node "0")
        terminal_pos: positive terminal node label
        terminal_neg: negative terminal node label

    Returns:
        dict with v_thevenin [V], r_thevenin [Ω], i_short_circuit [A],
        and step-by-step explanation list.
    """
    # ── Step 1: open-circuit voltage ──────────────────────────────────────
    oc_result = MNASolver(parse_netlist(netlist)).solve()
    if "error" in oc_result:
        return {"error": f"Open-circuit solve failed: {oc_result['error']}"}

    vp = oc_result["node_voltages"].get(terminal_pos, 0.0)
    vn = oc_result["node_voltages"].get(terminal_neg, 0.0)
    v_th = round(vp - vn, 9)

    # ── Step 2: short-circuit current via 0 V ammeter ─────────────────────
    sc_netlist = netlist.rstrip() + f"\nVsc {terminal_pos} {terminal_neg} 0"
    sc_result  = MNASolver(parse_netlist(sc_netlist)).solve()
    if "error" in sc_result:
        return {
            "v_thevenin": v_th,
            "error": f"Short-circuit solve failed: {sc_result['error']}",
            "note": "V_th obtained; R_th computation requires a short-circuit solve.",
        }

    i_sc = sc_result["branch_currents"].get("Vsc", 0.0)

    # ── Step 3: R_th = V_th / I_sc ────────────────────────────────────────
    if abs(i_sc) < 1e-15:
        r_th     = float("inf")
        r_th_str = "∞  (ideal current source / open Thevenin resistance)"
    else:
        r_th     = round(v_th / i_sc, 9)
        r_th_str = f"{r_th:.6g} Ω"

    i_norton = i_sc  # Norton current = short-circuit current

    return {
        "v_thevenin":       v_th,
        "r_thevenin":       r_th if r_th != float("inf") else None,
        "r_thevenin_str":   r_th_str,
        "i_short_circuit":  round(i_sc, 9),
        "i_norton":         round(i_norton, 9),
        "terminals":        f"{terminal_pos} (pos) / {terminal_neg} (neg)",
        "method":           "V_th via open-circuit; R_th = V_th / I_sc",
        "steps": [
            f"1. Open-circuit voltage:  V_oc = V({terminal_pos}) − V({terminal_neg})"
            f" = {vp:.4f} − {vn:.4f} = {v_th:.4f} V  → V_th = {v_th:.4f} V",
            f"2. Short-circuit current: insert 0 V source across terminals"
            f" → I_sc = {i_sc:.6g} A",
            f"3. Thevenin resistance:   R_th = V_th / I_sc"
            f" = {v_th:.4f} / {i_sc:.6g} = {r_th_str}",
            f"4. Norton equivalent:     I_N = {i_norton:.6g} A, R_N = R_th = {r_th_str}",
        ],
    }


# ─── Public API used by the tutor tool handler ───────────────────────────────

def solve_circuit(netlist: str) -> dict:
    """High-level entry point: parse netlist and return MNA solution."""
    try:
        components = parse_netlist(netlist)
        if not components:
            return {"error": "Netlist is empty or could not be parsed."}
        result = MNASolver(components).solve()
        # Strip verbose matrix-build log from the tool response to Claude
        result.pop("mna_build_steps", None)
        return result
    except Exception as exc:
        return {"error": f"Solver exception: {exc}"}


# ─── Quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Voltage divider: 10 V source, R1=4kΩ, R2=6kΩ → V(node 2) should be 6 V
    netlist = """
    V1 1 0 10
    R1 1 2 4k
    R2 2 0 6k
    """
    result = solve_circuit(netlist)
    print("Voltage divider test:")
    print(json.dumps(result, indent=2))
    print()

    th = compute_thevenin(netlist, "2", "0")
    print("Thevenin from node 2 to ground:")
    print(json.dumps(th, indent=2))
