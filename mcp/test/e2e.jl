# End-to-end check of the PHPS MCP server over its real transport: start the server, speak JSON-RPC on its standard
# input/output as a client would, build the base IEEE 39-bus model, run a short bus-16 fault, read the energy account
# and two signals. Run: julia --project=mcp mcp/test/e2e.jl   (the first build compiles the simulator: minutes)

import JSON3

const SERVER = joinpath(@__DIR__, "..", "server.jl")
proc = open(`$(Base.julia_cmd()) --project=$(joinpath(@__DIR__, "..")) --startup-file=no $SERVER`, "r+")
nextid = Ref(0)

function call(method, params = Dict{String,Any}(); notify = false)
    msg = Dict{String,Any}("jsonrpc" => "2.0", "method" => method, "params" => params)
    notify || (nextid[] += 1; msg["id"] = nextid[])
    println(proc, JSON3.write(msg)); flush(proc)
    notify && return nothing
    while true
        line = readline(proc)
        isempty(line) && eof(proc) && error("server closed its output")
        r = JSON3.read(line, Dict{String,Any})
        get(r, "id", nothing) == nextid[] && return r            # skip notifications (progress)
    end
end

tool(name, args = Dict{String,Any}()) = (r = call("tools/call", Dict{String,Any}("name" => name, "arguments" => args));
                                         haskey(r, "error") && error("$name: $(r["error"])");
                                         res = r["result"];
                                         txt = join([c["text"] for c in res["content"] if c["type"] == "text"], "\n");
                                         println("---- $name", get(res, "isError", false) ? " (ERROR)" : "", "\n", txt);
                                         (res, txt))

function main()
r = call("initialize", Dict{String,Any}("protocolVersion" => "2024-11-05", "capabilities" => Dict{String,Any}(),
                                        "clientInfo" => Dict("name" => "e2e", "version" => "0")))
println("initialize: ", r["result"]["serverInfo"])
call("notifications/initialized"; notify = true)
tl = call("tools/list")["result"]["tools"]
println("tools: ", join([t["name"] for t in tl], ", "))
@assert length(tl) == 13

tool("list_cases")
tool("build_model", Dict{String,Any}("preset" => "ieee", "dry_run" => true))
_, txt = tool("build_model", Dict{String,Any}("preset" => "ieee", "wait_seconds" => 30))
job = match(r"Job (j-[A-Za-z0-9-]+)", txt)[1]
while true
    _, txt = tool("job_status", Dict{String,Any}("job_id" => job, "wait_seconds" => 60))
    occursin(": done", txt) && break
    occursin("failed", txt) && error("build failed")
end
mid = match(r"Model (m-[A-Za-z0-9-]+) built", txt)[1]
tool("operating_point", Dict{String,Any}("model_id" => mid))
tool("describe", Dict{String,Any}("model_id" => mid, "component" => "IEEET1_2"))
# refusals that must come back as errors, not crashes
res, _ = tool("run_simulation", Dict{String,Any}("model_id" => mid, "tstop" => 0.3,
              "faults" => [Dict("bus" => 16, "kind" => "ag", "t_on" => 0.1, "t_off" => 0.15)]))
@assert get(res, "isError", false)                              # ground fault without the zero sequence
res, _ = tool("build_model", Dict{String,Any}("parameter_changes" => Dict("IEEET1_2" => Dict("bus" => 3))))
@assert get(res, "isError", false)                              # protected parameter
_, txt = tool("run_simulation", Dict{String,Any}("model_id" => mid, "tstop" => 0.25, "wait_seconds" => 60,
                                                 "faults" => [Dict("bus" => 16, "kind" => "abc", "R" => 0.001, "t_on" => 0.1, "t_off" => 0.15)]))
job = match(r"Job (j-[A-Za-z0-9-]+)", txt)[1]
while !occursin(": done", txt)
    occursin("failed", txt) && error("run failed")
    _, txt = tool("job_status", Dict{String,Any}("job_id" => job, "wait_seconds" => 60))
end
rid = match(r"Run (r-[A-Za-z0-9-]+) finished", txt)[1]
tool("list_signals", Dict{String,Any}("run_id" => rid))
tool("get_signals", Dict{String,Any}("run_id" => rid, "signals" => ["GENROU_3.speed", "bus16.voltage"], "max_points" => 40))
_, txt = tool("energy_audit", Dict{String,Any}("run_id" => rid))
res, _ = tool("plot", Dict{String,Any}("run_ids" => [rid], "signals" => ["bus16.voltage", "GENROU_3.speed"]))
@assert any(c -> c["type"] == "image", res["content"])
tool("list_runs")
close(proc)
println("\nE2E PASSED")
end
main()
