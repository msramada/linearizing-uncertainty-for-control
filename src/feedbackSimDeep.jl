Aa = [0.95;;]
Bb = [1.0;;]

K_lqr = lqr(Discrete, Aa, Bb, I, I)
Q_features_lqr = zero.(Ann)
Q_features_lqr[1:NinfoState,1:NinfoState] = I(NinfoState)
#K₁ = lqr(Discrete, Ā, B̄, Q_features_lqr, I)
K₁ = lqr(Discrete, Ann, Bnn, Q_features_lqr, I)
simHorizon = 600
x_true1 = zeros(n, simHorizon+1)
bbₖ = zeros(NinfoState, simHorizon+1)
bbₖ[:,1] = lₖ[:,1]
let x₁ = x₀
let Σ₁ = Σ₀
	for k in 1:simHorizon
		u₀ = - K₁ * encoderModel(bbₖ[:,k])
		#U_rec[:,k] = u₀
		x_true1[:,k+1] = eKF.next_state_sample(x_true1[:,k], u₀, dyna)
		y_true = eKF.output_sample(x_true1[:,k+1], dyna)
		Y_rec[:,k+1] = y_true
		Σ₊ = Σ₁
		x₁, Σ₁ = eKF.update(x₁, Σ₁, u₀, y_true, dyna)
		bbₖ[:,k+1] = make_info_state(x₁, Σ₊, dyna, infoType)
	end
end
end
estimErrDeep = mean((x_true1 - bbₖ[1:n,:]) .^ 2)
costDeep = LinearAlgebra.tr(bbₖ * bbₖ')
println("Experimental cost achieved by NN control: $costDeep")
println("Experimental estimation error: $estimErrDeep")
x_true2 = zeros(n, simHorizon+1)
bₖ = zeros(NinfoState, simHorizon+1)
bₖ[:,1] = lₖ[:,1]
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
costLQR = LinearAlgebra.tr(bₖ * bₖ')
estimErrLQR = mean((x_true2 - bₖ[1:n,:]) .^ 2)
println("Experimental cost achieved by lqr control: $costLQR")
println("Experimental estimation error: $estimErrLQR")
a1 = plot(bₖ[1,:], ylabel = L"$z_k$", label="LQR")
a1 = plot!(bbₖ[1,:], label="DeepK")

a2 = plot(bₖ[n+1,:], ylabel = L"tr", label="LQR")
a2 = plot!(bbₖ[n+1,:], label="DeepK")

#a3 = plot(bₖ[n+2,:], ylabel = L"tr", label="LQR")
#a3 = plot!(bbₖ[n+2,:], label="DeepK")

a4 = plot(x_true2[1,:], ylabel = L"x_{true}", label="LQR")
a4 = plot!(x_true1[1,:], label="DeepK")
title = "Costs: LQR: $costLQR, DeepK: $costDeep. Mean estim. errors: $estimErrLQR, $estimErrDeep"
#a4 = plot(bₖ[2,:], ylabel = L"\theta")
#a4 = plot!(lₖ[2,:])

savefig("figs/feedbackSimDeep.png")
display(plot(a1,a2,a4, layout=(3,1), xlabel = L"k"))



#=


=#






