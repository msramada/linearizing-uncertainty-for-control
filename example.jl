using Plots, ControlSystemsBase, StatsBase, LaTeXStrings
using LinearAlgebra, Zygote
include("src/eKF.jl")
include("src/info_feature_state.jl")

##### Pick chosen example ######
systemNumber = 1
include("./models4example.jl")

##### Define Dynamic System Object #####
dyna = eKF.StateSpaceSys(stateDynamics, outputDynamics, Q, R, Q_true)
n = dyna.n
#### Feature vector params #####
order = 1 # Highest degree of multinomial
make_feature = x -> [x;1.0]
horizon = 5_000
x_true = zeros(n, horizon+1)
x_true[:,1] = x₀ + sqrt(dyna.Q_true) * randn(n,)
#### Chose make_info_state: {1) cholesky: with_cholesky_Σ, 2) trace(QΣ): with_trace_QΣ}
infoType = "cholesky" # or trace

#### Collect features data #####
function sim2learn_eKF(x₀::Vector{Float64}, Σ₀::Matrix, U_rec::Matrix)
	η₀ = make_info_state([x₀;zero(x₀)], Σ₀, dyna, infoType)
	NinfoState = length(η₀)
	ηₖ = zeros(NinfoState, horizon+1)
	ηₖ[:,1] = η₀
	Nfeatures = length(make_feature(η₀))
	features = zeros(Nfeatures, horizon+1)
	features[:,1] = make_feature(ηₖ[:,1])
	println("Number of features is $Nfeatures.")
	println("Start collecting data ... ")
	x₁, Σ₁ = x₀, Σ₀
	for k in 1:horizon
		u₀ = U_rec[:,k]
		x_true[:,k+1] = eKF.next_state_sample(x_true[:,k], u₀, dyna)
		y_true = eKF.output_sample(x_true[:,k+1], dyna)
		xₚ, _ = eKF.time_update(x₁, Σ₁, u₀, dyna)
		y_true = dyna.h(xₚ)
		Σ₀ = Σ₁
		x₀ = x₁
		x₁, Σ₁ = eKF.update(x₁, Σ₁, u₀, y_true, dyna)
		ηₖ[:,k+1] = make_info_state([x₁;x₀], Σ₀, dyna, infoType)
		features[:,k+1] = make_feature(ηₖ[:,k+1])
	end
	return ηₖ, features, Nfeatures, NinfoState
end

U_rec = 0.2 * randn(1, horizon)
ηₖ, features, Nfeatures, NinfoState = sim2learn_eKF(x₀, Σ₀, U_rec)
println("Features data has been collected.")

########## Learning and Control ###########
include("src/Learn_and_Sim.jl")

learnt_features, System = learn_system(features, U_rec)
p1 = plot(features[2,1:200])
p1 = plot!(learnt_features[2,1:200])
p2 = plot(features[2n+1,1:200])
p2 = plot!(learnt_features[2n+1,1:200])
plot(p1,p2, layout=(2,1))
savefig("figs/learning_result.png")
simHorizon = 1000
x_DMD, x_lqr, Qₛₛ, x_true1, x_true2 = simulate_system(System, simHorizon, x₀, Σ₀)
Qₛₛ = Qₛₛ[1:end-1,1:end-1]

Result_cost1 = "Experimental cost achieved by lqr control: $(LinearAlgebra.tr(Qₛₛ * x_lqr * x_lqr'))"
Result_cost2 = "Experimental estimation error: $(mean((x_true2 - x_lqr[1:n,:]) .^ 2))"

Result_est_err1 = "Experimental cost achieved by eDMD control: $(LinearAlgebra.tr(Qₛₛ * x_DMD * x_DMD'))"
Result_est_err2 = "Experimental estimation error: $(mean((x_true1 - x_DMD[1:n,:]) .^ 2))"

print(Result_cost1,"\n",Result_cost2,"\n",Result_est_err1,"\n",Result_est_err2)
open("figs/paper_figs/results.txt", "w") do file
	println(file, Result_cost1,"\n",Result_cost2,"\n",Result_est_err1,"\n",Result_est_err2)
end

a1 = plot(x_lqr[1,:], ylabel = L"$z_k$", label="LQR")
a1 = plot!(x_DMD[1,:], label="DMDc")


a2 = plot(x_lqr[2n+1,:], ylabel = "uncertainty", label="LQR")
a2 = plot!(x_DMD[2n+1,:], label="DMDc")

a3 = plot(x_true2[1,:], ylabel = L"x_{true}", label="LQR")
a3 = plot!(x_true1[1,:], label="DMDc")

plot(a1,a2,a3, layout=(3,1))
savefig("figs/feedbackSim.png")


include("plotting_paper.jl")
