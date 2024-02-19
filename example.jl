using Plots, ControlSystemsBase, StatsBase, LaTeXStrings
using LinearAlgebra, Zygote, FileIO, JLD2
include("src/eKF.jl")
include("src/info_feature_state.jl")

##### Load example model ######
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
	η₀ = make_info_state(x₀, Σ₀, dyna, infoType)
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
		u₁ = U_rec[:,k]
		x_true[:,k+1] = eKF.next_state_sample(x_true[:,k], u₁, dyna)
		y_true = dyna.h(x₁) # during learning, yₖ - h(xₖ) = 0 => yₖ = h(xₖ)
		x₁, Σ₁ = eKF.update(x₁, Σ₁, u₁, y_true, dyna)
		ηₖ[:,k+1] = make_info_state(x₁, Σ₁, dyna, infoType)
		features[:,k+1] = make_feature(ηₖ[:,k+1])
	end
	return ηₖ, features, Nfeatures, NinfoState
end

U_rec = 0.5 * randn(1, horizon)
ηₖ, features, Nfeatures, NinfoState = sim2learn_eKF(x₀, Σ₀, U_rec)
println("Features data has been collected.")

########## Learning and Control ###########
include("src/Learn_and_Sim.jl")

learnt_features, System = learn_system(features, U_rec)

simHorizon = 1000
x_DMD, U_DMD, x_lqr, U_lqr, Qₛₛ, x_true1, x_true2 = simulate_system(System, simHorizon, x₀, Σ₀)
Qₛₛ = Qₛₛ[1:end-1,1:end-1]

function save_to_results()
	avg_cost_lqr = (LinearAlgebra.tr(Qₛₛ * x_lqr * x_lqr') + sum(U_lqr .^2))/simHorizon
	sse_lqr = mean((x_true2 - x_lqr[1:n,:]) .^ 2)
	Result_cost1 = "Experimental cost achieved by lqr control (per k): $avg_cost_lqr"
	Result_cost2 = "Experimental estimation error: $sse_lqr"
	avg_cost_DMD = (LinearAlgebra.tr(Qₛₛ * x_DMD * x_DMD') + sum(U_DMD .^2))/simHorizon
	sse_DMD = mean((x_true1 - x_DMD[1:n,:]) .^ 2)
	Result_est_err1 = "Experimental cost achieved by eDMD control (per k): $avg_cost_DMD, reduction: $((avg_cost_lqr-avg_cost_DMD)/avg_cost_lqr)"
	Result_est_err2 = "Experimental estimation error: $sse_DMD, reduction: $((sse_lqr - sse_DMD)/sse_lqr)"

	print(Result_cost1,"\n",Result_cost2,"\n",Result_est_err1,"\n",Result_est_err2)
	open("results/results.txt", "w") do file
		println(file, Result_cost1,"\n",
		Result_cost2,"\n",Result_est_err1,"\n",Result_est_err2)
	end
	return 1
end

save_to_results()


FileIO.save("results/ExampleData.jld2", "x_lqr", x_lqr, "x_DMD", x_DMD, "n", n, 
			"features", features, "learnt_features", learnt_features)
include("plotting_paper.jl")
