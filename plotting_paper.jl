using FileIO, JLD2, LaTeXStrings, Plots, Lux
x_lqr, x_DMD, n, features, learnt_features = FileIO.load("results/ExampleData.jld2", "x_lqr", "x_DMD",
                                            "n", "features", "learnt_features")
plt1 = plot(
                    xlim=(0,1000),
                    framestyle = :box,
                    yguidefontsize=8,
                    xguidefontsize=8,
                    xtickfontsize=6,
                    ytickfontsize=6,
					ylabel = L"$\sum_i^2 \hat x^i_{k\mid k}$",
                    palette = :seaborn_muted,
					xticks=:none,
                    legend=:none,
                    fontfamily="Computer Modern"
                )
plot!(plt1, x_lqr[1,:] .+ x_lqr[2,:])
plot!(plt1, x_DMD[1,:] .+ x_DMD[2,:])

plt2 = plot(
                    xlim=(0,1000),
					#ylim=(0,1),
                    framestyle = :box,
                    yguidefontsize=8,
                    xguidefontsize=8,
                    xtickfontsize=6,
                    ytickfontsize=6,
                    xlabel = L"k",
                    palette = :seaborn_muted,
					ylabel = L"$\Sigma^{1,1}_{k \mid k}$",
                    legend=:none,
                    fontfamily="Computer Modern"
                )
plot!(plt2, x_lqr[2n+1,:] .^ 2)
plot!(plt2, x_DMD[2n+1,:] .^ 2)

plot(plt1,plt2, layout=(2,1), size = (280,250))


savefig("figs/closed_loop_sim.pdf")



plt1 = plot(
                    xlim=(0,1000),
                    framestyle = :box,
                    yguidefontsize=8,
                    xguidefontsize=8,
                    xtickfontsize=6,
                    ytickfontsize=6,
					ylabel = L"$\hat x^1_{k\mid k}$",
                    palette = :seaborn_muted,
					xticks=:none,
                    legend=:none,
                    fontfamily="Computer Modern"
                )

plt2 = plot(
                    xlim=(0,1000),
                    framestyle = :box,
                    yguidefontsize=8,
                    xguidefontsize=8,
                    xtickfontsize=6,
                    ytickfontsize=6,
                    xlabel = L"k",
                    palette = :seaborn_muted,
					ylabel = L"$L_k^{1,1}$",
                    legend=:none,
                    fontfamily="Computer Modern"
                )

plot!(plt1, features[2,:], lw=2.0, size =(280,100))
plot!(plt1, learnt_features[2,:], lw=1.5, linestyle=:dot)
plot!(plt2, features[2n+1,:], lw=2.0, size =(280,100))
plot!(plt2, learnt_features[2n+1,:], lw =1.5, linestyle=:dot)


plot(plt1,plt2, layout=(2,1), size = (280,250))


savefig("figs/learning_result.pdf")


plt_ = plot(
                    size = (250,110),
                    framestyle = :box,
                    yguidefontsize=8,
                    xguidefontsize=8,
                    xtickfontsize=6,
                    ytickfontsize=6,
                    xlim = (-5,5),
                    xlabel = L"x",
					ylabel = L"ELU$(x)$",
                    legend=:none,
                    fontfamily="Computer Modern"
                )

plot!(plt_, x -> elu(x), lw = 1.5, font = 1, palette = :seaborn_muted)
savefig("figs/elu_fun.pdf")
