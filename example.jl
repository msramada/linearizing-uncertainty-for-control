using Plots, ControlSystemsBase, StatsBase, LaTeXStrings
using LinearAlgebra, Zygote
include("src/eKF.jl")
include("src/info_feature_state.jl")

##### Pick chosen example ######
systemNumber = 3
include("./models4example.jl")

##### Define Dynamic System Object #####
dyna = eKF.StateSpaceSys(stateDynamics, outputDynamics, Q, R, Q_true)
n = dyna.n

#### Feature vector params #####
delays =0 # Number of delays in the Hankel-based basis
order = 2 # Highest degree of multinomial
make_feature = x -> makeFeature1(x, order, dyna)
horizon = 10000
x_true = zeros(n, horizon+1)
x_true[:,1] = x₀ + sqrt(dyna.Q_true) * randn(n,)
U_rec = zeros(1, horizon)
Y_rec = zeros(1, horizon+1)

A = [0.95;;]
B = [1.0;;]

K_lqr = lqr(Discrete, A, B, I, I)

#### Chose make_info_state: {1) cholesky: with_cholesky_Σ, 2) trace(QΣ): with_trace_QΣ}
infoType = "cholesky" # or trace
l₀ = make_info_state(x₀, Σ₀, dyna, infoType)
NinfoState = length(l₀)
lₖ = zeros(NinfoState, horizon+1)
lₖ[:,1] = l₀
Nfeatures = length(make_feature(l₀))
features = zeros(Nfeatures, horizon+1)
features[:,1] = make_feature(lₖ[:,1])
println("Number of features is $Nfeatures.")
println("Start simulation ... ")
#### Collect features data #####
let x₁ = x₀
let Σ₁ = Σ₀
	for k in 1:horizon
		u₀ = sqrt(1.0) * randn(1,) - K_lqr * lₖ[1:1,k]
		U_rec[:,k] = u₀
		x_true[:,k+1] = eKF.next_state_sample(x_true[:,k], u₀, dyna)
		y_true = eKF.output_sample(x_true[:,k+1], dyna)
		Y_rec[:,k+1] = y_true
		#xₚ, _ = eKF.time_update(x₁, Σ₁, u₀, dyna)
		#y_true = dyna.h(xₚ)
		x₁, Σ₁ = eKF.update(x₁, Σ₁, u₀, y_true, dyna)
		lₖ[:,k+1] = make_info_state(x₁, Σ₁, dyna, infoType)
		features[:,k+1] = make_feature(lₖ[:,k+1])
	end
end
end
println("Features data has been collected.")

include("src/DMDc_truncation.jl")

include("src/feedbackSim.jl")
