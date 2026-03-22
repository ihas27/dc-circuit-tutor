"""
DC Circuit Tutor — Claude Opus 4.6 with adaptive thinking + tool use.

Claude selects the analysis method (KVL/KCL, nodal, mesh, superposition,
Thevenin/Norton), explains the approach step by step, then calls the Python
MNA solver via tools to compute exact numerical answers.
"""

import json
import sys
from typing import Optional
import anthropic
from solver import solve_circuit, compute_thevenin


# ─── Tool Definitions (JSON Schema) ──────────────────────────────────────────

TOOLS = [
    {
        "name": "solve_circuit",
        "description": (
            "Solve any DC circuit using Modified Nodal Analysis (MNA). "
            "Returns node voltages [V], branch currents [A], and power "
            "dissipation [W] for every element. Use this to compute "
            "exact numerical answers after explaining your method."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "netlist": {
                    "type": "string",
                    "description": (
                        "SPICE-like netlist, one component per line:\n"
                        "  <name> <node+> <node-> <value>  [# comment]\n\n"
                        "Supported component types (first letter of name):\n"
                        "  R  — resistor [Ω]   e.g. R1 1 2 1000  or  R1 1 2 1k\n"
                        "  V  — voltage source [V]   V1 1 0 12  (node 1 is +)\n"
                        "  I  — current source [A]   I1 0 2 0.002  (flows from 0 into 2)\n\n"
                        "SI prefixes: G M k m u n p  (e.g. 4.7k = 4700)\n"
                        "Ground node is always labelled '0'."
                    ),
                }
            },
            "required": ["netlist"],
        },
    },
    {
        "name": "compute_thevenin",
        "description": (
            "Compute the Thevenin (and Norton) equivalent of a circuit "
            "between two specified terminals. "
            "Returns V_th [V], R_th [Ω], I_sc [A], and step-by-step workings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "netlist": {
                    "type": "string",
                    "description": "Same SPICE-like netlist format as solve_circuit.",
                },
                "terminal_pos": {
                    "type": "string",
                    "description": "Positive terminal node label (e.g. '2').",
                },
                "terminal_neg": {
                    "type": "string",
                    "description": "Negative terminal node label (e.g. '0' for ground).",
                },
            },
            "required": ["netlist", "terminal_pos", "terminal_neg"],
        },
    },
]


# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a DC circuit analysis tutor for undergraduate engineering students
(ECE / EE programs). Your job is to teach — not just compute.

## Your workflow for each circuit problem

1. **Understand the circuit** — describe the topology (series, parallel, bridge, etc.)
   and identify the source types present.

2. **Select a method** — choose the best analysis technique and justify WHY:
   - KVL / KCL          → simple series-parallel, few loops/nodes
   - Nodal Analysis      → few nodes, current sources present, or need all voltages
   - Mesh Analysis       → few meshes, voltage sources dominate
   - Superposition       → multiple independent sources (⚠️ invalid with dependent sources)
   - Thevenin / Norton   → finding equivalent circuit for a load

3. **Set up equations manually** — walk through the steps as you'd write them on paper:
   write out the KVL/KCL equations, set up the nodal matrix, or define meshes.

4. **Use the tools** — call `solve_circuit` or `compute_thevenin` to compute the
   exact numerical answer and verify your hand analysis.

5. **Interpret results** — explain what the numbers mean physically: voltage drops,
   current paths, power balance, energy efficiency.

## Netlist format for the tools
```
R1 1 2 1000    # 1 kΩ resistor between nodes 1 and 2
V1 1 0 12      # 12 V source, node 1 positive, ground is node 0
I1 0 3 0.002   # 2 mA source flowing from node 0 into node 3
R2 2 0 4.7k    # 4.7 kΩ to ground (note SI prefix 'k')
```
Ground is always node "0". Node labels can be numbers or names (e.g. "A", "out").

## Tone
Be engaging and educational. Show your reasoning. Flag common mistakes
(floating nodes, sign conventions, superposition with dependent sources, etc.).
Use formatted equations when helpful.
"""


# ─── Tool Execution ───────────────────────────────────────────────────────────

def _execute_tool(name: str, inputs: dict) -> str:
    """Call the Python solver and return a JSON string result."""
    if name == "solve_circuit":
        result = solve_circuit(inputs["netlist"])
    elif name == "compute_thevenin":
        result = compute_thevenin(
            inputs["netlist"],
            inputs["terminal_pos"],
            inputs["terminal_neg"],
        )
    else:
        result = {"error": f"Unknown tool: '{name}'"}
    return json.dumps(result, indent=2)


def _print_tool_summary(name: str, result_json: str) -> None:
    """Print a concise one-liner summarising tool output."""
    try:
        data = json.loads(result_json)
    except Exception:
        return

    if name == "solve_circuit" and "node_voltages" in data:
        vstr = ", ".join(
            f"V({n})={v:.4g} V"
            for n, v in sorted(data["node_voltages"].items())
            if n != "0"
        )
        print(f"  ↳ {vstr}")
        if data.get("total_supplied_W") is not None:
            print(f"  ↳ Power supplied = {data['total_supplied_W']:.4g} W")

    elif name == "compute_thevenin" and "v_thevenin" in data:
        print(
            f"  ↳ V_th = {data['v_thevenin']:.4g} V, "
            f"R_th = {data.get('r_thevenin_str', 'N/A')}, "
            f"I_sc = {data.get('i_short_circuit', 'N/A'):.4g} A"
        )


# ─── Main Tutor Loop ──────────────────────────────────────────────────────────

def run_tutor(circuit_description: str, verbose: bool = True) -> str:
    """
    Run the tutoring session for a circuit problem.

    Uses Claude Opus 4.6 with:
      - adaptive thinking  (model decides how much to reason internally)
      - streaming          (text appears as it is generated)
      - tool use loop      (Python solver called for exact computation)

    Args:
        circuit_description: Free-text problem or structured netlist prompt.
        verbose:             If True, stream text to stdout.

    Returns:
        The full concatenated text response from Claude.
    """
    client   = anthropic.Anthropic()
    messages = [{"role": "user", "content": circuit_description}]
    full_text_parts: list[str] = []

    if verbose:
        width = 64
        print("\n" + "═" * width)
        print("  DC Circuit Tutor  ·  Claude Opus 4.6  ·  Adaptive Thinking")
        print("═" * width + "\n")

    # ── Agentic tool-use loop ──────────────────────────────────────────────
    while True:
        with client.messages.stream(
            model      = "claude-opus-4-6",
            max_tokens = 8192,
            thinking   = {"type": "adaptive"},
            system     = SYSTEM_PROMPT,
            tools      = TOOLS,
            messages   = messages,
        ) as stream:

            in_thinking_block = False

            for event in stream:
                if event.type == "content_block_start":
                    btype = getattr(event.content_block, "type", None)
                    if btype == "thinking":
                        in_thinking_block = True
                        if verbose:
                            print("  [thinking…]\n", flush=True)
                    elif btype == "text":
                        in_thinking_block = False

                elif event.type == "content_block_stop":
                    in_thinking_block = False

                elif event.type == "content_block_delta" and not in_thinking_block:
                    if event.delta.type == "text_delta":
                        chunk = event.delta.text
                        if verbose:
                            print(chunk, end="", flush=True)
                        full_text_parts.append(chunk)

            response = stream.get_final_message()

        if verbose:
            print()  # newline after streamed block

        # ── Collect any tool-use requests ──────────────────────────────────
        tool_blocks = [b for b in response.content if b.type == "tool_use"]

        if response.stop_reason == "end_turn" or not tool_blocks:
            break

        # ── Execute tools and feed results back ────────────────────────────
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tb in tool_blocks:
            if verbose:
                print(f"\n  [tool: {tb.name}]", flush=True)
            result_str = _execute_tool(tb.name, tb.input)
            if verbose:
                _print_tool_summary(tb.name, result_str)
                print()
            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": tb.id,
                "content":     result_str,
            })

        messages.append({"role": "user", "content": tool_results})

    return "".join(full_text_parts)
