using Lux, Random, Optimisers, Distributions, Statistics
plotting_horizon = 200
η = 1e-5
liftedDim = 32
N_EPOCHS = 20_000
rng = Random.default_rng()
Random.seed!(rng, 0)
Nhidden = 32
Nencoder = liftedDim
BATCH_SIZE = 64
ActivationFunc = relu
Φₑ = Chain(
	Dense(NinfoState, Nhidden, ActivationFunc),
	Dense(Nhidden, Nhidden, ActivationFunc),
	Dense(Nhidden, Nhidden, ActivationFunc),
#	Dense(Nhidden, Nhidden, ActivationFunc),
	Dense(Nhidden, Nencoder-NinfoState)
	)
function encoder_concat(Encoder_NN, state, p, ls)
	return [state; Encoder_NN(state, p, ls)[1]]
end

params, ls = Lux.setup(rng, Φₑ)
TrainingHorizon = floor(Int, size(lₖ)[2]/1.2)
TrainingData = lₖ[:,1:TrainingHorizon]
ValidationData = lₖ[:,TrainingHorizon+1:end]
A = randn(liftedDim, liftedDim)
B = randn(liftedDim, 1)

function loss_fn(p, p₀, ls, states_list, controls_list, A, B)
	pred_horizon = length(states_list)
	gₓ = encoder_concat(Φₑ, states_list[1], p₀, ls)
	Kgₓ = [A B] * [gₓ; controls_list[1]]
	gₓ₊ = encoder_concat(Φₑ, states_list[2], p, ls)
	PredictionLoss = 1 * mean((Kgₓ .- gₓ₊) .^ 2)
	for j in 2:pred_horizon-1
		Kgₓ = [A B] * [Kgₓ;controls_list[j]]
		gₓ₊ = encoder_concat(Φₑ, states_list[j+1], p, ls)
		PredictionLoss += 1 * mean((Kgₓ .- gₓ₊) .^ 2)
	end
	return PredictionLoss / pred_horizon
end

opt = Optimisers.Adam(η)
opt_state = Optimisers.setup(opt, params)
learning_pred_horizon = 2
LS_N = 5000

function Train(ps, opt_state, A, B)
avg_loss = 0.0
bb = η
α = 0
for epoch=1:N_EPOCHS
	indices = rand(1:TrainingHorizon - learning_pred_horizon, BATCH_SIZE)
	states_list = []
	controls_list = []
	for j in 0:learning_pred_horizon-1
		push!(states_list, TrainingData[:,indices .+ j])
		push!(controls_list, U_rec[:,indices .+ j])
	end
	Loss, Back, = Zygote.pullback(p -> loss_fn(p, ps,
			ls, states_list, controls_list, A, B), ps)
	grad, = Back(1.0)
	opt_state, ps = Optimisers.update(opt_state, ps, grad)
	avg_loss += Loss / N_EPOCHS

if epoch % 1000 == 0
	println("$(100 * round(epoch/N_EPOCHS; digits=3))%",
		" complete ... Loss is $(round(avg_loss, digits=5))")
	avg_loss = 0.0
	#bb *= 0.97
	#opt = Optimisers.Adam(bb)
	#opt_state = Optimisers.setup(opt, ps)
	X = encoder_concat(Φₑ, TrainingData[:,1:LS_N], ps, ls)
	ScalingVec = sum(X, dims = 2) ./ TrainingHorizon
	S = inv(LinearAlgebra.Diagonal(ScalingVec[:]))
	S = I
	#X⁺, _ = Φₑ(TrainingData[:,2:end], ps.layer_1, ls.layer_1)
	X⁺ = X[:,2:end]
	X = X[:,1:end-1]
	X, X⁺ = S * X, S * X⁺
	Γ = X⁺ / [X; U_rec[:,1:LS_N-1]]
	A = inv(S) * Γ[:,1:liftedDim] * S
	B = inv(S) * Γ[:,liftedDim+1:end]
	xTrunc0 = inv(S) * X[:,1]
	X = inv(S) * X
	PredFeaturesDeep = ls_lsim(A, B, 0, [U_rec 0], feature0=xTrunc0)
	a1=plot(TrainingData[1,1:plotting_horizon])
	a1 = plot!(PredFeaturesDeep[1,1:plotting_horizon])
	b1=plot(TrainingData[n+1,1:plotting_horizon])
	b1 = plot!(PredFeaturesDeep[n+1,1:plotting_horizon])
	c1=plot(TrainingData[n+2,1:plotting_horizon])
	c1 = plot!(PredFeaturesDeep[n+2,1:plotting_horizon])
	display(plot(a1,b1,c1, layout=(3,1)))
end
end
return ps, A, B
end

ps, Ann, Bnn = Train(params, opt_state, A, B)

println("Learning has been done!")
encoderModel = state -> encoder_concat(Φₑ, state, ps, ls)
savefig("figs/DeepKoop.png")

#=
X, _ = Φₑ(ValidationData[:,1:end-1], ps.layer_1, ls.layer_1)
X, _ = Φd(X, ps.layer_2, ls.layer_2)
a1=plot(ValidationData[1,1:plotting_horizon])
a1=plot!(X[1,1:plotting_horizon])
b1=plot(ValidationData[n+1,1:plotting_horizon])
b1=plot!(X[n+1,1:plotting_horizon])
ab = plot(a1,b1, layout=(2,1))
display(ab)
model = Φₑ
ps = ps.layer_1
ls = ls.layer_1
xTrunc0 = X[:,1]
PredFeaturesDeep = ls_lsim(A, B, 0, [U_rec 0], feature0=xTrunc0)
println(PredFeaturesDeep[1,10])
kk = 1
p1 = plot(lₖ[kk,1:plotting_horizon])
p1 = plot!(PredFeaturesDeep[kk,1:plotting_horizon])

kk = n+1
p2 = plot(lₖ[kk,1:plotting_horizon])
p2 = plot!(PredFeaturesDeep[kk,1:plotting_horizon])



plot(p1,p2, layout=(2,1), reuse = false)

#savefig("DeepKoop.png")
=#