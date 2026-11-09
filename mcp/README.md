# PHPS MCP server

Lets an AI assistant (Claude Code or any Model Context Protocol client) drive the PHPS EMT simulator: build the
IEEE 39-bus model with chosen options, look at its operating point, apply faults, run, and read waveforms, the energy
account and plots. The assistant calls fixed tools with checked arguments; it never runs code through the server.

## Use

Registered in the repository's `.mcp.json` as `phps`. Claude Code started in the repository root offers it after
approval. By hand:

```
julia --project=mcp --startup-file=no mcp/server.jl
```

First setup (once): `julia --project=mcp -e 'using Pkg; Pkg.instantiate()'`.

## Tools

`list_cases`, `build_model`, `operating_point`, `describe`, `run_simulation`, `job_status`, `cancel_job`, `list_runs`,
`list_signals`, `get_signals`, `energy_audit`, `plot`, `delete_run`. Each tool's arguments are described in the tool
list the client receives.

## How it works

- `server.jl` speaks the protocol on standard input/output and loads no simulator code.
- `worker.jl` is one persistent Julia process with the simulator; it builds and runs one job at a time, gets a fresh
  model for every run, and writes its output to `outputs/mcp/worker.log`.
- Builds and runs are jobs: a tool waits up to `wait_seconds`, then returns a job id for `job_status`.
- A run longer than `max_wall_minutes` (default 10) by its estimate starts only with `allow_long = true`; the limit is
  also enforced while it runs.
- Everything is stored under `outputs/mcp/`: `models/<id>/` (the derived case, the build options, the operating
  point) and `runs/<id>/` (`run.jld2` with the states, bus voltages, energy account and recorded generator signals;
  `run.json`; plots). Signals, the energy account and plots are read from these files, never simulated again.

End-to-end test: `mcp/test/e2e.jl`.
