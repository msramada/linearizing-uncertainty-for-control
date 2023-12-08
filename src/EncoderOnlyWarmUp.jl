using Lux, Random, Optimisers, Distributions, Statistics
plotting_horizon = 200
η = 1e-3
liftedDim = Nfeatures
N_EPOCHS = 5_000
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

function loss_warmup(X, p, ls)
	Xₑ = Φₑ(X, p, ls)[1]
	X_dot = [X[:,i] for i in 1:size(X)[2]]

	feach = make_feature.(X_dot)

	Matfeach = reduce(hcat, feach)
	println(Matfeach)
	loss = 1 * mean((Matfeach - [X; Xₑ]) .^ 2)
	return loss
end

function warm_up(ps, ls, TrainingData)
opt = Optimisers.Adam(η)
opt_state = Optimisers.setup(opt, ps)	
avg_loss = 0.0
	for epoch in 1:200
		indices = rand(1:TrainingHorizon, BATCH_SIZE)
		X = TrainingData[:,indices]	
		Loss, Back, = Zygote.pullback(p -> loss_warmup(X, p, ls), ps)
		grad, = Back(1.0)
		opt_state, ps = Optimisers.update(opt_state, ps, grad)
		avg_loss += Loss / N_EPOCHS
		if epoch % 20 == 0
			println("Warm-up learning $(100 * round(epoch/200; digits=3))%",
				", loss is $avg_loss")
			avg_loss = 0.0
		end
		
	end
	return ps
end


ValidationData = lₖ[:,TrainingHorizon+1:end]
A = randn(liftedDim, liftedDim)
B = randn(liftedDim, 1)

function loss_fn(p, ls, states_list, controls_list, A, B)
	pred_horizon = length(states_list)
	gₓ = encoder_concat(Φₑ, states_list[1], p, ls)
	Kgₓ = [A B] * [gₓ; controls_list[1]]
	gₓ₊ = encoder_concat(Φₑ, states_list[2], p, ls)
	PredictionLoss = 1 * sum((Kgₓ .- gₓ₊) .^ 2)
	for j in 2:pred_horizon-1
		Kgₓ = [A B] * [Kgₓ;controls_list[j]]
		gₓ₊ = encoder_concat(Φₑ, states_list[j+1], p, ls)
		PredictionLoss += 1 * sum((Kgₓ .- gₓ₊) .^ 2)
	end
	return PredictionLoss
end

opt = Optimisers.Adam(η)
opt_state = Optimisers.setup(opt, params)
learning_pred_horizon = 3
LS_N = 500

function Train(ps, opt_state, A, B)
avg_loss = 0.0
bb = η
for epoch in 1:N_EPOCHS
	indices = rand(1:TrainingHorizon - learning_pred_horizon, BATCH_SIZE)
	states_list = []
	controls_list = []
	for j in 0:learning_pred_horizon-1
		push!(states_list, TrainingData[:,indices .+ j])
		push!(controls_list, U_rec[:,indices .+ j])
	end
	Loss, Back, = Zygote.pullback(p -> loss_fn(p,
			ls, states_list, controls_list, A, B), ps)
	grad, = Back(1.0)
	opt_state, ps = Optimisers.update(opt_state, ps, grad)
	avg_loss += Loss / N_EPOCHS

if epoch % 100 == 1
	println("$(100 * round(epoch/N_EPOCHS; digits=3))%",
		" complete ... Loss is $(round(avg_loss, digits=5))")
	avg_loss = 0.0
	#bb *= 0.97
	#opt = Optimisers.Adam(bb)
	#opt_state = Optimisers.setup(opt, ps)
	X = encoder_concat(Φₑ, TrainingData[:,1:LS_N], ps, ls)
	#X = [TrainingData[:,1:LS_N]; Φₑ(TrainingData[:,1:LS_N], ps, ls)[1]]
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
	a1=plot!(X[1,1:plotting_horizon])
	a1 = plot!(PredFeaturesDeep[1,1:plotting_horizon])
	b1=plot(TrainingData[n+1,1:plotting_horizon])
	b1 = plot!(PredFeaturesDeep[n+1,1:plotting_horizon])
	c1=plot(X[NinfoState+3,1:plotting_horizon])
	c1 = plot!(PredFeaturesDeep[NinfoState+3,1:plotting_horizon])
	display(plot(a1,b1,c1, layout=(3,1)))
end
end
return ps, A, B
end

params = warm_up(params, ls, TrainingData)
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