# PHPS documentation

| Page | What it covers |
|---|---|
| [Getting started](getting_started.md) | Installation, the first EMT and RMS runs, where results go, running the tests |
| [EMT simulation](emt_simulation.md) | Building a port-Hamiltonian EMT model, its options, faults and their clearing, running it, the energy account, stored runs |
| [RMS simulation](rms_simulation.md) | Loading a case and a scenario, power flow and equilibrium, the IDA and BDF1 solvers, results and run records |
| [Case files](case_format.md) | The system, scenario and full-model JSON formats |
| [Analysis tools](analysis.md) | Port-Hamiltonian audits, port passivity, interval enclosures, Lyapunov candidates and certificates, JuMP programs |
| [Scripts](../scripts/README.md) | The command-line entry points |
| [MCP server](../mcp/README.md) | Driving PHPS from an AI assistant |
| [API reference](api/README.md) | Every exported function and type, with its docstring, generated from the code |

The API reference is generated: after changing docstrings or exports, run

```
julia --project=. scripts/build_api_docs.jl
```

From the Julia prompt, `?name` shows the same docstring, for example `?phs_simulate`.
