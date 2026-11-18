# network_gen.jl: written by PHPS.RT.network_rt_generate (src/rt/network/network.jl); do not edit.
# The board top of the network block for h = 864e-9 s (NETWORK.md 4-5): three phase loops per step on
# 8 lanes x 2 axes; one shared multiply function (16 multipliers) serves all three; the lanes' programs are the
# ROM tables NET_* (network_rom.h); formats in the export (types); scale2(v, k) = v 2^k exactly (wiring in C++).

# the 16 products of one cycle: 8 lanes x (alpha, beta), one coefficient per lane (shared by all phase loops)
@hls function network_mac!(c::C[8], oa::X[8], ob::X[8], pa::P[8], pb::P[8])
    @pragma "HLS INLINE off"
    @pragma "HLS PIPELINE II=1"
    for l in 1:8
        @pragma "HLS UNROLL"
        pa[l] = c[l] * oa[l]
        pb[l] = c[l] * ob[l]
    end
end

@hls function network_run!(x::X[158], ph::X[2], n0::Int, n_on::Int, n_off::Int, nsteps::Int, decim::Int,
                           v1_out::Out{Sig}, v2_out::Out{Sig}, strobe::Out{Bool}, sample::Out{Bool})
    @pragma "HLS INTERFACE mode=s_axilite port=return"
    @pragma "HLS INTERFACE mode=s_axilite port=x storage_impl=bram"
    @pragma "HLS INTERFACE mode=s_axilite port=ph"
    @pragma "HLS INTERFACE mode=s_axilite port=n0"
    @pragma "HLS INTERFACE mode=s_axilite port=n_on"
    @pragma "HLS INTERFACE mode=s_axilite port=n_off"
    @pragma "HLS INTERFACE mode=s_axilite port=nsteps"
    @pragma "HLS INTERFACE mode=s_axilite port=decim"
    @pragma "HLS INTERFACE mode=ap_none port=v1_out"
    @pragma "HLS INTERFACE mode=ap_none port=v2_out"
    @pragma "HLS INTERFACE mode=ap_none port=strobe"
    @pragma "HLS INTERFACE mode=ap_none port=sample"
    xa::X[9, 12]                                # x operand banks, alpha (bank = writing lane; 9: the phasor)
    @pragma "HLS ARRAY_PARTITION variable=xa dim=1 complete"
    @pragma "HLS BIND_STORAGE variable=xa type=ram_1wnr impl=lutram"
    xb::X[9, 12]                                # beta
    @pragma "HLS ARRAY_PARTITION variable=xb dim=1 complete"
    @pragma "HLS BIND_STORAGE variable=xb type=ram_1wnr impl=lutram"
    xo::X[8, 12]                               # the owner's copy of its x slots (read-modify-write), so
    @pragma "HLS ARRAY_PARTITION variable=xo dim=1 complete"                                        # the x banks have one write and the lanes' reads only
    @pragma "HLS BIND_STORAGE variable=xo type=ram_1wnr impl=lutram"
    yo::X[8, 12]
    @pragma "HLS ARRAY_PARTITION variable=yo dim=1 complete"
    @pragma "HLS BIND_STORAGE variable=yo type=ram_1wnr impl=lutram"
    qa::X[8, 5]                              # rq banks (written in P1, read in P2)
    @pragma "HLS ARRAY_PARTITION variable=qa dim=1 complete"
    @pragma "HLS BIND_STORAGE variable=qa type=ram_1wnr impl=lutram"
    qb::X[8, 5]
    @pragma "HLS ARRAY_PARTITION variable=qb dim=1 complete"
    @pragma "HLS BIND_STORAGE variable=qb type=ram_1wnr impl=lutram"
    da::X[8, 5]                              # dq banks (written in P2, read in P3)
    @pragma "HLS ARRAY_PARTITION variable=da dim=1 complete"
    @pragma "HLS BIND_STORAGE variable=da type=ram_1wnr impl=lutram"
    db::X[8, 5]
    @pragma "HLS ARRAY_PARTITION variable=db dim=1 complete"
    @pragma "HLS BIND_STORAGE variable=db type=ram_1wnr impl=lutram"
    pa::X[8, 8]                               # rp, lane-local (written in P1, the init of P3)
    @pragma "HLS ARRAY_PARTITION variable=pa dim=1 complete"
    @pragma "HLS BIND_STORAGE variable=pa type=ram_1wnr impl=lutram"
    pb::X[8, 8]
    @pragma "HLS ARRAY_PARTITION variable=pb dim=1 complete"
    @pragma "HLS BIND_STORAGE variable=pb type=ram_1wnr impl=lutram"
    pn::X[8, 2]                                     # the phasor rows' increments, lane-local
    @pragma "HLS ARRAY_PARTITION variable=pn dim=0 complete"
    pm::X[8, 2]
    @pragma "HLS ARRAY_PARTITION variable=pm dim=0 complete"
    cc::C[8]                                        # one cycle's coefficients, operands and products
    @pragma "HLS ARRAY_PARTITION variable=cc complete"
    va::X[8]
    @pragma "HLS ARRAY_PARTITION variable=va complete"
    vb::X[8]
    @pragma "HLS ARRAY_PARTITION variable=vb complete"
    pa_::P[8]
    @pragma "HLS ARRAY_PARTITION variable=pa_ complete"
    pb_::P[8]
    @pragma "HLS ARRAY_PARTITION variable=pb_ complete"
    aa::Acc[8]                                      # accumulators (>= 16 guard bits below X's grid)
    @pragma "HLS ARRAY_PARTITION variable=aa complete"
    ab::Acc[8]
    @pragma "HLS ARRAY_PARTITION variable=ab complete"
    zero::Acc = 0
    oa1::X[25]                                       # the 25 bank outputs (1-8 x, 9 phasor, 10-17 rq, 18-25 dq)
    @pragma "HLS ARRAY_PARTITION variable=oa1 complete"                                   # of each lane group (lanes 1-4: port 1, 5-8: port 2)
    ob1::X[25]
    @pragma "HLS ARRAY_PARTITION variable=ob1 complete"
    oa2::X[25]
    @pragma "HLS ARRAY_PARTITION variable=oa2 complete"
    ob2::X[25]
    @pragma "HLS ARRAY_PARTITION variable=ob2 complete"
    rcp::C[8, 89] = rom(:NET_CP)               # the lanes' program, one entry per lane and cycle: the
    rcf::C[8, 89] = rom(:NET_CF)               # coefficient (pre-fault, fault), the source bank (1-25),
    rs::Ib[8, 89] = rom(:NET_RS)               # first / last entry of a row, the init from rp, the write
    rb1::Ib[25, 89] = rom(:NET_BA1)             # kind (1 rq, 2 rp, 3 phasor, 4 dq and x_q, 5 x_p) and its
    rb2::Ib[25, 89] = rom(:NET_BA2)             # slots; per bank and group one read address (rb1, rb2;
    rf::Bool[8, 89] = rom(:NET_F)               # unused: the parking slot)
    re::Bool[8, 89] = rom(:NET_E)
    rik::Bool[8, 89] = rom(:NET_IK)
    rwk::Ib[8, 89] = rom(:NET_WK)
    rdq::Ib[8, 89] = rom(:NET_DQ)
    rdp::Ib[8, 89] = rom(:NET_DP)
    rdd::Ib[8, 89] = rom(:NET_DD)
    rdn::Ib[8, 89] = rom(:NET_DN)
    rw::Ib[8, 89] = rom(:NET_W)
    ri::Ib[8, 89] = rom(:NET_I)
    cbk::Ib[79] = rom(:NET_CB)                       # bank and slot of each state (copy in and out)
    csl::Ib[79] = rom(:NET_CS)
    @pragma "HLS ARRAY_PARTITION variable=rcp dim=1 complete"
    @pragma "HLS ARRAY_PARTITION variable=rcf dim=1 complete"
    @pragma "HLS ARRAY_PARTITION variable=rs dim=1 complete"
    @pragma "HLS ARRAY_PARTITION variable=rf dim=1 complete"
    @pragma "HLS ARRAY_PARTITION variable=re dim=1 complete"
    @pragma "HLS ARRAY_PARTITION variable=rik dim=1 complete"
    @pragma "HLS ARRAY_PARTITION variable=rwk dim=1 complete"
    @pragma "HLS ARRAY_PARTITION variable=rdq dim=1 complete"
    @pragma "HLS ARRAY_PARTITION variable=rdp dim=1 complete"
    @pragma "HLS ARRAY_PARTITION variable=rdd dim=1 complete"
    @pragma "HLS ARRAY_PARTITION variable=rdn dim=1 complete"
    @pragma "HLS ARRAY_PARTITION variable=rw dim=1 complete"
    @pragma "HLS ARRAY_PARTITION variable=ri dim=1 complete"
    @pragma "HLS ARRAY_PARTITION variable=rb1 dim=1 complete"
    @pragma "HLS ARRAY_PARTITION variable=rb2 dim=1 complete"
    z0::X = 0                                        # the parking slots hold zero (read, never written)
    for l in 1:9
        xa[l, 12] = z0
        xb[l, 12] = z0
    end
    for l in 1:8
        xo[l, 12] = z0
        yo[l, 12] = z0
        qa[l, 5] = z0
        qb[l, 5] = z0
        da[l, 5] = z0
        db[l, 5] = z0
        pa[l, 8] = z0
        pb[l, 8] = z0
    end
    for r in 1:79                                    # copy in: the states to their banks (and owner copies)
        xa[cbk[r], csl[r]] = x[r]
        xb[cbk[r], csl[r]] = x[79 + r]
        xo[cbk[r], csl[r]] = x[r]
        yo[cbk[r], csl[r]] = x[79 + r]
    end
    xa[9, 1] = ph[1]                                 # the phasor bank: slot 1 (c, s), slot 2 (-s, c), x 2^sig
    xa[9, 2] = -ph[2]
    xb[9, 1] = ph[2]
    xb[9, 2] = ph[1]
    st::Bool = false
    cnt::Int = 0
    for i in 1:nsteps
        @pragma "HLS ALLOCATION function instances=network_mac limit=1"
        n::Int = n0 + i
        fault::Bool = false                          # the fault bank, latched for the whole step
        if n > n_on && n <= n_off
            fault = true
        end
        for t in 1:89                            # the step: rq, rp, phasor, dq (x_q), x_p rows interleaved
            @pragma "HLS PIPELINE II=1"
            @pragma "HLS DEPENDENCE variable=xa type=inter dependent=false"
            @pragma "HLS DEPENDENCE variable=xb type=inter dependent=false"
            @pragma "HLS DEPENDENCE variable=xo type=inter dependent=false"
            @pragma "HLS DEPENDENCE variable=yo type=inter dependent=false"
            @pragma "HLS DEPENDENCE variable=qa type=inter dependent=false"
            @pragma "HLS DEPENDENCE variable=qb type=inter dependent=false"
            @pragma "HLS DEPENDENCE variable=da type=inter dependent=false"
            @pragma "HLS DEPENDENCE variable=db type=inter dependent=false"
            @pragma "HLS DEPENDENCE variable=pa type=inter dependent=false"
            @pragma "HLS DEPENDENCE variable=pb type=inter dependent=false"
            for b in 1:9                             # one read per bank and group (review 37)
                @pragma "HLS UNROLL"
                oa1[b] = xa[b, rb1[b, t]]
                ob1[b] = xb[b, rb1[b, t]]
                oa2[b] = xa[b, rb2[b, t]]
                ob2[b] = xb[b, rb2[b, t]]
            end
            for b in 1:8
                @pragma "HLS UNROLL"
                oa1[9 + b] = qa[b, rb1[9 + b, t]]
                ob1[9 + b] = qb[b, rb1[9 + b, t]]
                oa2[9 + b] = qa[b, rb2[9 + b, t]]
                ob2[9 + b] = qb[b, rb2[9 + b, t]]
                oa1[17 + b] = da[b, rb1[17 + b, t]]
                ob1[17 + b] = db[b, rb1[17 + b, t]]
                oa2[17 + b] = da[b, rb2[17 + b, t]]
                ob2[17 + b] = db[b, rb2[17 + b, t]]
            end
            for l in 1:8
                @pragma "HLS UNROLL"
                cc[l] = rcp[l, t]
                if fault
                    cc[l] = rcf[l, t]
                end
                if l <= 4
                    va[l] = oa1[rs[l, t]]
                    vb[l] = ob1[rs[l, t]]
                else
                    va[l] = oa2[rs[l, t]]
                    vb[l] = ob2[rs[l, t]]
                end
            end
            network_mac!(cc, va, vb, pa_, pb_)
            for l in 1:8
                @pragma "HLS UNROLL"
                if rf[l, t]
                    if rik[l, t]
                        aa[l] = scale2(pa[l, ri[l, t]], 0) + pa_[l]
                        ab[l] = scale2(pb[l, ri[l, t]], 0) + pb_[l]
                    else
                        aa[l] = zero + pa_[l]
                        ab[l] = zero + pb_[l]
                    end
                else
                    aa[l] = aa[l] + pa_[l]
                    ab[l] = ab[l] + pb_[l]
                end
                if rwk[l, t] == 1
                    qa[l, rdq[l, t]] = scale2(aa[l], 2)
                    qb[l, rdq[l, t]] = scale2(ab[l], 2)
                elseif rwk[l, t] == 2
                    pa[l, rdp[l, t]] = scale2(aa[l], 7)
                    pb[l, rdp[l, t]] = scale2(ab[l], 7)
                elseif rwk[l, t] == 3
                    pn[l, rdn[l, t]] = scale2(aa[l], -11)
                    pm[l, rdn[l, t]] = scale2(ab[l], -11)
                elseif rwk[l, t] == 4
                    da[l, rdd[l, t]] = scale2(aa[l], 1)
                    db[l, rdd[l, t]] = scale2(ab[l], 1)
                    ya2::X = xo[l, rw[l, t]] + scale2(aa[l], -2)
                    yb2::X = yo[l, rw[l, t]] + scale2(ab[l], -2)
                    xo[l, rw[l, t]] = ya2
                    yo[l, rw[l, t]] = yb2
                    xa[l, rw[l, t]] = ya2
                    xb[l, rw[l, t]] = yb2
                elseif rwk[l, t] == 5
                    ya3::X = xo[l, rw[l, t]] + scale2(aa[l], -11)
                    yb3::X = yo[l, rw[l, t]] + scale2(ab[l], -11)
                    xo[l, rw[l, t]] = ya3
                    yo[l, rw[l, t]] = yb3
                    xa[l, rw[l, t]] = ya3
                    xb[l, rw[l, t]] = yb3
                end
            end
        end
        xa[9, 1] = xa[9, 1] + pn[8, 1]       # the phasor for the next step (P2, P3 do not read it)
        xa[9, 2] = xa[9, 2] + pn[1, 2]
        xb[9, 1] = xb[9, 1] + pm[8, 1]
        xb[9, 2] = xb[9, 2] + pm[1, 2]
        st = !st
        cnt += 1
        if cnt >= decim
            cnt = 0
            sample[] = true
        else
            sample[] = false
        end
        v1_out[] = xa[4, 2]      # alpha charge of node bus 16 (v = Q x), to the ILA
        v2_out[] = xa[1, 4]      # alpha charge of node bus 39
        strobe[] = st
    end
    for r in 1:79                                    # copy out
        x[r] = xa[cbk[r], csl[r]]
        x[79 + r] = xb[cbk[r], csl[r]]
    end
    ph[1] = xa[9, 1]
    ph[2] = xb[9, 1]
end
