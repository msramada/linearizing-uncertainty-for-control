plt1 = plot(
                    xlim=(0,1000),
                    framestyle = :box,
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
                    xlabel = L"k",
					ylabel = L"$\sqrt{\Sigma_{k \mid k}}$",
                    legend=:none,
                    fontfamily="Computer Modern"
                )
plot!(plt2, x_lqr[2n+1,:])
plot!(plt2, x_DMD[2n+1,:])

plot(plt1,plt2, layout=(2,1), size = (320,240), dpi=500)


savefig("figs/paper_figs/closed_loop_sim.png")



plt1 = plot(
                    xlim=(0,400),
                    framestyle = :box,
					ylabel = L"$\hat x_{k\mid k}$",
					xticks=:none,
                    legend=:none,
                    fontfamily="Computer Modern"
                )

plt2 = plot(
                    xlim=(0,400),
                    framestyle = :box,
                    xlabel = L"k",
					ylabel = L"$\sqrt{\Sigma_{k \mid k}}$",
                    legend=:none,
                    fontfamily="Computer Modern"
                )

plot!(plt1, features[2,:], lw=2.0)
plot!(plt1, learnt_features[2,:], lw=1.0)#, linestyle=:dashdot)
plot!(plt2, features[2n+1,:], lw=2.0)
plot!(plt2, learnt_features[2n+1,:])#, linestyle=:dashdot)


plot(plt1,plt2, layout=(2,1), size = (320,240), dpi=500)


savefig("figs/paper_figs/learning_result.png")


plt_ = plot(
                    size = (320,120),
                    framestyle = :box,
                    xlim = (-2,2),
                    xlabel = L"x",
					ylabel = L"elu$(x)$",
                    legend=:none,
                    fontfamily="Computer Modern"
                )

plot!(plt_, x -> elu(3 .* (x .- 1)), lw = 1.5)
savefig("figs/paper_figs/elu_fun.png")
