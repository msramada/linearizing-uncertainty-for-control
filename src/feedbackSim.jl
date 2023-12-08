A = [0.95;;]
B = [1.0;;]

K_lqr = lqr(Discrete, A, B, I, I)
Q_features_lqr = LinearAlgebra.diagm(zeros(size(System.A)[1],))
Q_features_lqr[1:NinfoState,1:NinfoState] = I(NinfoState)
#K₁ = lqr(Discrete, Ā, B̄, Q_features_lqr, I)
#Q_features_lqr = I
QQ = zeros(liftedDim, liftedDim)
QQ[1:NinfoState,1:NinfoState] = I(NinfoState)
QQ[2:n,2:n] = zero.(QQ[2:n,2:n])
QQ[n+2:NinfoState,n+2:NinfoState] = zero.(QQ[n+2:NinfoState,n+2:NinfoState])
QQQ = QQ[1:NinfoState,1:NinfoState]
K₁ = lqr(System, QQ, I)
simHorizon = 200
l₀ = make_info_state(x₀, Σ₀, dyna, infoType)
x_true1 = zeros(n, simHorizon+1)
bbₖ = zeros(NinfoState, simHorizon+1)
bbₖ[:,1] = l₀
let x₁ = x₀
let Σ₁ = Σ₀
	for k in 1:simHorizon
		u₀ = - K₁ * Û' * S * make_feature(bbₖ[:,k])
		#U_rec[:,k] = u₀
		x_true1[:,k+1] = eKF.next_state_sample(x_true1[:,k], u₀, dyna)
		y_true = eKF.output_sample(x_true1[:,k+1], dyna)
		Y_rec[:,k+1] = y_true
		x₁, Σ₁ = eKF.update(x₁, Σ₁, u₀, y_true, dyna)
		bbₖ[:,k+1] = make_info_state(x₁, Σ₁, dyna, infoType)
	end
end
end
println("Experimental cost achieved by eDMD control: $(LinearAlgebra.tr(QQQ * bbₖ * bbₖ'))")
println("Experimental estimation error: $(mean((x_true1 - bbₖ[1:n,:]) .^ 2))")

x_true2 = zeros(n, simHorizon+1)
bₖ = zeros(NinfoState, simHorizon+1)
bₖ[:,1] = l₀
let x₁ = x₀
let Σ₁ = Σ₀
	for k in 1:simHorizon
		u₀ = - K_lqr * bₖ[1:1,k]
		#U_rec[:,k] = u₀
		x_true2[:,k+1] = eKF.next_state_sample(x_true2[:,k], u₀, dyna)
		y_true = eKF.output_sample(x_true2[:,k+1], dyna)
		Y_rec[:,k+1] = y_true
		x₁, Σ₁ = eKF.update(x₁, Σ₁, u₀, y_true, dyna)
		bₖ[:,k+1] = make_info_state(x₁, Σ₁, dyna, infoType)
	end
end
end
println("Experimental cost achieved by lqr control: $(LinearAlgebra.tr(QQQ * bₖ * bₖ'))")
println("Experimental estimation error: $(mean((x_true2 - bₖ[1:n,:]) .^ 2))")
a1 = plot(bₖ[1,:], ylabel = L"$z_k$", label="LQR")
a1 = plot!(bbₖ[1,:], label="DMDc")

a2 = plot(bₖ[n+1,:], ylabel = L"tr", label="LQR")
a2 = plot!(bbₖ[n+1,:], label="DMDc")

a3 = plot(x_true2[1,:], ylabel = L"x_{true}", label="LQR")
a3 = plot!(x_true1[1,:], label="DMDc")

#a4 = plot(bₖ[2,:], ylabel = L"\theta")
#a4 = plot!(lₖ[2,:])

savefig("figs/feedbackSim.png")
display(plot(a1,a2,a3, layout=(3,1), xlabel = L"k"))