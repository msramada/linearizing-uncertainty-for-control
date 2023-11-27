A = [0.95;;]
B = [1.0;;]

K_lqr = lqr(Discrete, A, B, I, I)
Q_features_lqr = LinearAlgebra.diagm(zeros(size(SysTruncated1.A)[1],))
Q_features_lqr[1:NinfoState,1:NinfoState] = I(NinfoState)
K₁ = lqr(SysTruncated1, Q_features_lqr, I)
simHorizon = 200
l₀ = make_info_state(x₀, Σ₀, dyna, infoType)
x_true1 = zeros(n, simHorizon+1)
lₖ = zeros(NinfoState, simHorizon+1)
lₖ[:,1] = l₀
let x₁ = x₀
let Σ₁ = Σ₀
	for k in 1:simHorizon
		u₀ = - K₁ * Û' * S * make_feature(lₖ[:,k])
		println(u₀)
		#U_rec[:,k] = u₀
		x_true1[:,k+1] = eKF.next_state_sample(x_true1[:,k], u₀, dyna)
		y_true = eKF.output_sample(x_true1[:,k+1], dyna)
		Y_rec[:,k+1] = y_true
		#xₚ, _ = eKF.time_update(x₁, Σ₁, u₀, dyna)
		#y_true = dyna.h(xₚ)
		x₁, Σ₁ = eKF.update(x₁, Σ₁, u₀, y_true, dyna)
		lₖ[:,k+1] = make_info_state(x₁, Σ₁, dyna, infoType)
	end
end
end


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
		#xₚ, _ = eKF.time_update(x₁, Σ₁, u₀, dyna)
		#y_true = dyna.h(xₚ)
		x₁, Σ₁ = eKF.update(x₁, Σ₁, u₀, y_true, dyna)
		bₖ[:,k+1] = make_info_state(x₁, Σ₁, dyna, infoType)
	end
end
end

a1 = plot(bₖ[1,:], ylabel = L"z_k")
a1 = plot!(lₖ[1,:])

a2 = plot(bₖ[3,:], ylabel = L"tr")
a2 = plot!(lₖ[3,:])

#a3 = plot(x_true2[1,:], ylabel = L"x_{true}")
#a3 = plot!(x_true1[1,:])

#a4 = plot(bₖ[2,:], ylabel = L"\theta")
#a4 = plot!(lₖ[2,:])

plot(a1,a2, layout=(2,1), reuse = false, xlabel = L"k")