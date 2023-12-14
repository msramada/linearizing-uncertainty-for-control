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
liftedDim = 4
#### Feature vector params #####
delays = 2 # Number of delays in the Hankel-based basis
order = 1 # Highest degree of multinomial
make_feature = x -> makeFeature_Trigon(x, order, dyna)
make_feature = x -> [x;1.0]
horizon = 5_000
x_true = zeros(n, horizon+1)
x_true[:,1] = x₀ + sqrt(dyna.Q_true) * randn(n,)
A = [0.95;;]
B = [1.0;;]

K_lqr = lqr(Discrete, A, B, I, I)

#### Chose make_info_state: {1) cholesky: with_cholesky_Σ, 2) trace(QΣ): with_trace_QΣ}
infoType = "cholesky" # or trace

#### Collect features data #####
function sim2learn_eKF(x₀::Vector{Float64}, Σ₀::Matrix, U_rec)
	l₀ = make_info_state(x₀, Σ₀, dyna, infoType)
	NinfoState = length(l₀)
	lₖ = zeros(NinfoState, horizon+1)
	lₖ[:,1] = l₀
	Nfeatures = length(make_feature(l₀))
	features = zeros(Nfeatures, horizon+1)
	features[:,1] = make_feature(lₖ[:,1])
	println("Number of features is $Nfeatures.")
	println("Start collecting data ... ")
	x₁, Σ₁ = x₀, Σ₀
	for k in 1:horizon
		u₀ = U_rec[:,k]
		x_true[:,k+1] = eKF.next_state_sample(x_true[:,k], u₀, dyna)
		y_true = eKF.output_sample(x_true[:,k+1], dyna)
		xₚ, _ = eKF.time_update(x₁, Σ₁, u₀, dyna)
		y_true = dyna.h(xₚ)
		Σ₊ = Σ₁
		x₁, Σ₁ = eKF.update(x₁, Σ₁, u₀, y_true, dyna)
		lₖ[:,k+1] = make_info_state(x₁, Σ₊, dyna, infoType)
		features[:,k+1] = make_feature(lₖ[:,k+1])
	end
	return lₖ, features, Nfeatures, NinfoState
end

U_rec = 0.2 * randn(1, horizon)
lₖ, features, Nfeatures, NinfoState = sim2learn_eKF(x₀, Σ₀, U_rec)
println("Features data has been collected.")

#include("src/DMDc_truncation.jl")
include("src/withPseudoInv.jl")
#include("src/feedbackSim.jl")
