"""
DC Circuit Tutor — CLI entry point.

Usage:
    python main.py                           # interactive mode
    python main.py --example voltage_divider
    python main.py --example wheatstone
    python main.py --example thevenin
    python main.py --example superposition
    python main.py "your circuit description here"
    python main.py --test                    # run solver unit tests (no API needed)
"""

import argparse
import sys
import json


# ─── Built-in example circuit problems ───────────────────────────────────────

EXAMPLES: dict[str, str] = {

    "voltage_divider": """\
Solve this voltage divider circuit step by step:

  • V1 = 12 V source (node 1 positive, node 0 is ground)
  • R1 = 4 kΩ between nodes 1 and 2
  • R2 = 6 kΩ between node 2 and ground

Find:
  (a) The voltage at node 2
  (b) Current through each resistor
  (c) Power dissipated in each resistor
  (d) Verify power balance (supplied = dissipated)

Show which analysis method you choose and why.
""",

    "wheatstone": """\
Analyze this Wheatstone bridge and determine if it is balanced:

  • V1 = 10 V source (node 1 positive, node 0 ground)
  • R1 = 100 Ω  from node 1 to node 2
  • R2 = 100 Ω  from node 1 to node 3
  • R3 = 100 Ω  from node 2 to node 4
  • R4 = 120 Ω  from node 3 to node 4
  • R5 = 50 Ω   galvanometer from node 2 to node 3
  • Node 4 connected to ground

Questions:
  (a) What current flows through the galvanometer R5?
  (b) Is the bridge balanced? What resistance would balance it?
  (c) What is the equivalent resistance seen by V1?
""",

    "thevenin": """\
Find the Thevenin equivalent of the following circuit as seen from nodes A and ground:

  • V1 = 15 V  (node 1 positive, node 0 ground)
  • R1 = 3 kΩ  from node 1 to node A
  • R2 = 6 kΩ  from node A to ground
  • R3 = 2 kΩ  from node 1 to ground

Show:
  (a) V_th (open-circuit voltage at node A)
  (b) R_th (Thevenin resistance seen from node A)
  (c) The Norton equivalent
  (d) If a 12 kΩ load is connected to the Thevenin equivalent, what is the load voltage?
""",

    "superposition": """\
Use the superposition principle to find the current through R2 in this circuit:

  • V1 = 12 V source from node 1 to ground
  • I1 = 2 mA current source flowing from ground into node 2
  • R1 = 6 kΩ from node 1 to node 2
  • R2 = 3 kΩ from node 2 to ground

Steps to show:
  (a) Contribution of V1 alone (I1 deactivated — open circuit)
  (b) Contribution of I1 alone (V1 deactivated — short circuit)
  (c) Total current through R2 by superposition
  (d) Verify with nodal analysis of the complete circuit
""",
}


# ─── Solver self-test (no API required) ──────────────────────────────────────

def run_self_tests() -> None:
    """Run numerical unit tests on the MNA solver."""
    from solver import solve_circuit, compute_thevenin

    PASS = "\033[92m✓\033[0m"
    FAIL = "\033[91m✗\033[0m"
    errors = 0

    def check(label: str, got: float, expected: float, tol: float = 1e-6) -> None:
        nonlocal errors
        ok = abs(got - expected) < tol
        symbol = PASS if ok else FAIL
        print(f"  {symbol}  {label}: got {got:.6g}, expected {expected:.6g}")
        if not ok:
            errors += 1

    print("\n── Solver Unit Tests ──────────────────────────────────────")

    # Test 1: Voltage divider
    print("\n[1] Voltage divider  (V1=10V, R1=4kΩ, R2=6kΩ)")
    r = solve_circuit("V1 1 0 10\nR1 1 2 4k\nR2 2 0 6k")
    check("V(node 2)", r["node_voltages"]["2"], 6.0, tol=1e-4)
    check("I(R1)",     r["branch_currents"]["R1"],  1e-3, tol=1e-8)
    check("P(R1) W",   r["power_resistors_W"]["R1"], 4e-3, tol=1e-8)
    check("Power balance",
          r["total_supplied_W"], r["total_dissipated_W"], tol=1e-7)

    # Test 2: Two current sources, one resistor
    print("\n[2] Current divider  (I1=1mA into node1, I2=3mA into node1, R1=2kΩ to gnd)")
    r2 = solve_circuit("I1 0 1 0.001\nI2 0 1 0.003\nR1 1 0 2000")
    check("V(node 1)", r2["node_voltages"]["1"], 8.0, tol=1e-4)

    # Test 3: Thevenin
    print("\n[3] Thevenin  (V1=10V, R1=4kΩ, R2=6kΩ — from node 2 to gnd)")
    th = compute_thevenin("V1 1 0 10\nR1 1 2 4k\nR2 2 0 6k", "2", "0")
    check("V_th",   th["v_thevenin"],     6.0,    tol=1e-4)
    check("R_th",   th["r_thevenin"],     2400.0, tol=1e-3)
    check("I_sc",   th["i_short_circuit"], 6.0 / 2400.0, tol=1e-8)

    # Test 4: Superposition (V1=12V, I1=2mA, R1=6kΩ, R2=3kΩ)
    print("\n[4] Mixed source circuit  (V1=12V, I1=2mA, R1=6kΩ node1→2, R2=3kΩ node2→gnd)")
    r4 = solve_circuit("V1 1 0 12\nI1 0 2 0.002\nR1 1 2 6000\nR2 2 0 3000")
    # Nodal at node 2: (12-V2)/6k + 2m = V2/3k  →  V2 = (12/6k + 2m) / (1/6k + 1/3k)
    v2_expected = (12.0 / 6000 + 0.002) / (1.0 / 6000 + 1.0 / 3000)
    check("V(node 2)", r4["node_voltages"]["2"], v2_expected, tol=1e-4)

    print()
    if errors == 0:
        print(f"  {PASS}  All tests passed.\n")
    else:
        print(f"  {FAIL}  {errors} test(s) failed.\n")
        sys.exit(1)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DC Circuit Solver & Tutor — Claude Opus 4.6 + Python MNA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Examples:",
            "  python main.py --example voltage_divider",
            "  python main.py --example thevenin",
            "  python main.py --test",
            "  python main.py 'Find V across R2: 9V source, R1=3k, R2=6k in series'",
        ]),
    )
    parser.add_argument(
        "circuit",
        nargs="?",
        help="Circuit description (natural language). Omit to enter interactively.",
    )
    parser.add_argument(
        "--example",
        choices=list(EXAMPLES.keys()),
        metavar="NAME",
        help=f"Run a built-in example. Choices: {', '.join(EXAMPLES)}",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run solver unit tests (no Claude API call needed).",
    )
    args = parser.parse_args()

    if args.test:
        run_self_tests()
        return

    if args.example:
        description = EXAMPLES[args.example]
        print(f"Example: {args.example}")
    elif args.circuit:
        description = args.circuit
    else:
        # ── Interactive mode ───────────────────────────────────────────────
        print("DC Circuit Tutor")
        print("─" * 40)
        print(f"Built-in examples: {', '.join(EXAMPLES)}")
        print("Run with:  python main.py --example <name>\n")
        print("Or describe your circuit below.")
        print("(Press Enter twice to submit)\n")
        lines: list[str] = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        description = "\n".join(lines).strip()
        if not description:
            print("No input provided.")
            sys.exit(1)

    from tutor import run_tutor
    run_tutor(description)


if __name__ == "__main__":
    main()
