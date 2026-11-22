# Listing 3 of the paper (candidate): a component declared once in the template; its rates, energy account and port
# powers are generated from the declaration. Checked against the simulator's own branch row (tpl_branch_row).
# Run from the PHPS folder: julia --project=. scripts/paper/listing3_component.jl
using PHPS
import PHPS: PHTemplateComponent, tpl_states,
             tpl_energy, tpl_gradient, tpl_domain, tpl_block
# a series R-L branch in the dq frame (a complex pair)
struct RLBranch <: PHTemplateComponent
    R::Float64; L::Float64; wb::Float64
end
tpl_states(::RLBranch) = ["phi_re", "phi_im"]  # fluxes
tpl_energy(c::RLBranch, x) = c.wb * sum(abs2, x) / 2c.L
tpl_gradient(c::RLBranch, x) = c.wb .* x ./ c.L  # current
tpl_domain(::RLBranch, x) = nothing
function tpl_block(c::RLBranch, x, ext)  # [phi; v_a; v_b]
    J = zeros(6, 6); R = zeros(6, 6)
    J[1, 2] = c.L; J[2, 1] = -c.L        # frame rotation
    for k in 1:2
        J[k, 2+k] = 1;  J[2+k, k] = -1   # from node a
        J[k, 4+k] = -1; J[4+k, k] = 1    # to node b
    end
    R[1, 1] = R[2, 2] = c.R
    (; J, R, G = zeros(6, 0), v = Float64[],
       sources = String[], ports = [:v_a, :v_b],
       port_dims = [2, 2])
end
b = RLBranch(0.01, 0.1, 2pi * 60)
x, ext = [1e-4, -2e-4], [1.0, 0.0, 0.98, -0.05]
r = PHPS.tpl_rows(b, x, ext)     # rates, generated
a = PHPS.tpl_account(b, x, ext)  # energy, loss, port power
# --- checks (not part of the printed listing)
i = complex(PHPS.tpl_gradient(b, x)...)
ref = PHPS.tpl_branch_row(b.R, b.L, 1.0 + 0im, false, i, complex(ext[1], ext[2]), complex(ext[3], ext[4]))
println("rates: ", r.dx, "  simulator's branch row: ", [real(ref), imag(ref)], "  difference ", maximum(abs.(r.dx .- [real(ref), imag(ref)])))
println("account: H = ", a.energy, ", loss = ", a.loss, ", port power = ", a.port_power,
        ";  H' - (port power - loss) = ", PHPS.tpl_rate(b, x, r.dx) - (a.port_power - a.loss))
