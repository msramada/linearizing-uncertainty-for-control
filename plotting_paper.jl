# Used solely to generate the figures used in our paper.
using FileIO, JLD2, LaTeXStrings, Plots, Lux, Measures
x_lqr, x_DMD, n, features, learnt_features = FileIO.load("results/ExampleData.jld2", "x_lqr", "x_DMD",
                                            "n", "features", "learnt_features")

width = 280
height = 170
plt1 = plot(
                    xlim=(0,1000),
                    framestyle = :box,
                    yguidefontsize=7,
                    xguidefontsize=7,
                    xtickfontsize=6,
                    ytickfontsize=6,
					ylabel = L"$\sum_i^3 x^i_{k\mid k}$",
                    palette = :seaborn_muted,
					xticks=:none,
                    legend=:none,
                    fontfamily="Computer Modern"
                )
plot!(plt1, x_lqr[1,:] .+ x_lqr[2,:] .+ x_lqr[3,:])
plot!(plt1, x_DMD[1,:] .+ x_DMD[2,:] .+ x_DMD[3,:],xformatter=_->"")

function calc_Σ_tr(vec::Matrix{Float64})
    trace = zeros(size(vec)[2],)
    for i=1:size(vec)[2]
        trace[i] = tr(back_from_cholesky_Σ(vec[:,i], dyna)[2])
    end
    return trace
end

plt2 = plot(
                    xlim=(0,1000),
					#ylim=(0,1),
                    framestyle = :box,
                    yguidefontsize=7,
                    xguidefontsize=7,
                    xtickfontsize=6,
                    ytickfontsize=6,
                    xlabel = L"k",
                    palette = :seaborn_muted,
					ylabel = L"tr($\Sigma_{k \mid k-1}$)",
                    legend=:none,
                    fontfamily="Computer Modern"
                )
plot!(plt2, calc_Σ_tr(x_lqr))
plot!(plt2, calc_Σ_tr(x_DMD))

plot(plt1,plt2, layout=(2,1), bottom_margin = [-4mm 0mm], size = (width,height))


savefig("figs/closed_loop_sim.pdf")


xlim2 = 500
plt1 = plot(
                    xlim=(0,xlim2),
                    framestyle = :box,
                    yguidefontsize=7,
                    xguidefontsize=7,
                    xtickfontsize=6,
                    ytickfontsize=6,
					ylabel = L"$L_k^{(1,1)}$",
                    xlabel = L"k",
                    palette = :seaborn_muted,
					#xticks=:none,
                    legend=:none,
                    fontfamily="Computer Modern"
                )

plot!(plt1, features[n+1,:], lw=2.0)
plot!(plt1, learnt_features[n+1,:], lw=1.5, linestyle=:dot)

plot(plt1, size = (width,height/2))

savefig("figs/learning_result.pdf")

plt1 = plot(
                    xlim=(0,300),
                    framestyle = :box,
                    yguidefontsize=7,
                    xguidefontsize=7,
                    xtickfontsize=6,
                    ytickfontsize=6,
					ylabel = L"$\sum_i^3 x^i_{k\mid k}$",
                    palette = :seaborn_muted,
					xticks=:none,
                    legend=:none,
                    fontfamily="Computer Modern"
                )
plot!(plt1, x_lqr[1,:] .+ x_lqr[2,:] .+ x_lqr[3,:])
plot!(plt1, sum(x_true2, dims=1)[:])

plt2 = plot(
                    xlim=(0,300),
					#ylim=(0,1),
                    framestyle = :box,
                    yguidefontsize=7,
                    xguidefontsize=7,
                    xtickfontsize=6,
                    ytickfontsize=6,
                    xlabel = L"k",
                    palette = :seaborn_muted,
					ylabel = L"tr($\Sigma_{k \mid k-1}$)",
                    legend=:none,
                    fontfamily="Computer Modern"
                )
plot!(plt2, x_DMD[1,:] .+ x_DMD[2,:] .+ x_DMD[3,:])
plot!(plt2, sum(x_true1, dims=1)[:])

plot(plt1,plt2, layout=(2,1), bottom_margin = [-4mm 0mm], size = (width,height))


savefig("figs/compared_to_true.pdf")


plt1 = plot(
                    xlim=(100,200),
                    ylim=(-8,8),
                    framestyle = :box,
                    yguidefontsize=7,
                    xguidefontsize=7,
                    xtickfontsize=6,
                    ytickfontsize=6,
					ylabel = "magnitude",
                    xlabel = L"k",
                    palette = :seaborn_muted,
                    legend=:none,
                    fontfamily="Computer Modern"
                )
plot!(plt1, x_lqr[1,:] .+ x_lqr[2,:] .+ x_lqr[3,:], color="blue")
plot!(plt1, sum(x_true2, dims=1)[:], color="blue", linestyle=:dot)

plot!(plt1, x_DMD[1,:] .+ x_DMD[2,:] .+ x_DMD[3,:], color="black")
plot!(plt1, sum(x_true1, dims=1)[:], color="black", linestyle=:dot)

plot(plt1, size = (width,height/2))


savefig("figs/compared_to_true.pdf")


