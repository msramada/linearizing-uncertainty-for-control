using FileIO, JLD2, LaTeXStrings, Plots, Lux
x_lqr, x_DMD, n, features, learnt_features = FileIO.load("ExampleData.jld2", "x_lqr", "x_DMD",
                                            "n", "features", "learnt_features")
plt1 = plot(
                    xlim=(0,1000),
                    framestyle = :box,
                    yguidefontsize=8,
                    xguidefontsize=8,
                    xtickfontsize=6,
                    ytickfontsize=6,
					ylabel = L"$\hat x_{k\mid k}$",
					xticks=:none,
                    legend=:none,
                    fontfamily="Computer Modern"
                )
plot!(plt1, x_lqr[1,:])
plot!(plt1, x_DMD[1,:])

plt2 = plot(
                    xlim=(0,1000),
					ylim=(0,1),
                    framestyle = :box,
                    yguidefontsize=8,
                    xguidefontsize=8,
                    xtickfontsize=6,
                    ytickfontsize=6,
                    xlabel = L"k",
					ylabel = L"$\sqrt{\Sigma_{k \mid k}}$",
                    legend=:none,
                    fontfamily="Computer Modern"
                )
plot!(plt2, x_lqr[2n+1,:])
plot!(plt2, x_DMD[2n+1,:])

plot(plt1,plt2, layout=(2,1), size = (250,200))


savefig("figs/paper_figs/closed_loop_sim.pdf")



plt1 = plot(
                    xlim=(0,200),
                    framestyle = :box,
                    yguidefontsize=8,
                    xguidefontsize=8,
                    xtickfontsize=6,
                    ytickfontsize=6,
					ylabel = L"$\hat x_{k\mid k}$",
					xticks=:none,
                    legend=:none,
                    fontfamily="Computer Modern"
                )

plt2 = plot(
                    xlim=(0,200),
                    framestyle = :box,
                    yguidefontsize=8,
                    xguidefontsize=8,
                    xtickfontsize=6,
                    ytickfontsize=6,
                    xlabel = L"k",
					ylabel = L"$\sqrt{\Sigma_{k \mid k}}$",
                    legend=:none,
                    fontfamily="Computer Modern"
                )

plot!(plt1, features[2,:], lw=2.0)
plot!(plt1, learnt_features[2,:], lw=1.0)#, linestyle=:dashdot)
plot!(plt2, features[2n+1,:], lw=2.0)
plot!(plt2, learnt_features[2n+1,:])#, linestyle=:dashdot)


plot(plt1,plt2, layout=(2,1), size = (250,200))


savefig("figs/paper_figs/learning_result.pdf")


plt_ = plot(
                    size = (250,110),
                    framestyle = :box,
                    yguidefontsize=8,
                    xguidefontsize=8,
                    xtickfontsize=6,
                    ytickfontsize=6,
                    xlim = (-2,2),
                    xlabel = L"x",
					ylabel = L"ELU$(3(x-1))$",
                    legend=:none,
                    fontfamily="Computer Modern"
                )

plot!(plt_, x -> elu(3 .* (x .- 1)), lw = 1.5, font = 1)
savefig("figs/paper_figs/elu_fun.pdf")
