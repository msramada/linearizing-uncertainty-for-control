using Lux, Random, Optimisers, Distributions, Statistics
plotting_horizon = 200
η = 0.0001
N_EPOCHS = 10_000
rng = Random.default_rng()
Random.seed!(rng, 0)
Nhidden = 128
Nencoder = liftedDim
BATCH_SIZE = 32
ActivationFunc = relu
Φₑ = Chain(
	Dense(NinfoState, Nhidden, ActivationFunc),
	Dense(Nhidden, Nhidden, ActivationFunc),
	Dense(Nhidden, Nhidden, ActivationFunc),
#	Dense(Nhidden, Nhidden, ActivationFunc),
	Dense(Nhidden, Nencoder)
	)
Φd = Chain(
	Dense(Nencoder, Nhidden, ActivationFunc),
	Dense(Nhidden, Nhidden, ActivationFunc),
	Dense(Nhidden, Nhidden, ActivationFunc),
#	Dense(Nhidden, Nhidden, ActivationFunc),
	Dense(Nhidden, NinfoState)
	)
Φₖ = Parallel(nothing, Φₑ, Φd)

params, ls = Lux.setup(rng, Φₖ)
TrainingHorizon = floor(Int, size(lₖ)[2]/2)
TrainingData = lₖ[:,1:TrainingHorizon]
ValidationData = lₖ[:,TrainingHorizon+1:end]
A = randn(liftedDim, liftedDim)
B = randn(liftedDim, 1)
function loss_fn(p, ls, states_list, controls_list, A, B)
	pred_horizon = length(states_list)
	states = states_list[1]
	u₁ = controls_list[1]
	gₓ, _ = Φₑ(states, p.layer_1, ls.layer_1)
	states_from_gₓ, _ = Φd(gₓ, p.layer_2, ls.layer_2)
	Kgₓ = [A B] * [gₓ;controls_list[1]]
	states_from_Kgₓ, _ = Φd(Kgₓ, p.layer_2, ls.layer_2)
	TransformLoss = 1 * mean((states .- states_from_gₓ) .^2)
	gₓ₊, _ = Φₑ(states_list[2], p.layer_1, ls.layer_1)	
	PredictionLoss = 1 * mean((Kgₓ .- gₓ₊) .^2) 
	
	for j in 2:pred_horizon-1
		Kgₓ = [A B] * [Kgₓ;controls_list[j]]
		gₓ₊, _ = Φₑ(states_list[j+1], p.layer_1, ls.layer_1)	
		PredictionLoss += 1 * mean((Kgₓ .- gₓ₊) .^2)
		states_from_gₓ₊, _ = Φd(gₓ₊, p.layer_2, ls.layer_2)
		states_from_Kgₓ, _ = Φd(Kgₓ, p.layer_2, ls.layer_2)
		PredictionLoss += 1 * mean((states_from_gₓ .- states_from_Kgₓ) .^2) 
	end
	return PredictionLoss + TransformLoss
end

opt = Optimisers.Adam(η)
opt_state = Optimisers.setup(opt, params)
learning_pred_horizon = 2

function Train(ps, opt_state, A, B)
avg_loss = 0.0
for epoch=1:N_EPOCHS
	indices = rand(1:TrainingHorizon - learning_pred_horizon, BATCH_SIZE)
	states_list = []
	controls_list = []
	for j in 0:learning_pred_horizon-1
		push!(states_list, TrainingData[:,indices .+ j])
		push!(controls_list, U_rec[:,indices .+ j])
	end
	Loss = loss_fn(ps, ls, states_list, controls_list, A, B)
	avg_loss += Loss / N_EPOCHS
	grad  = Zygote.gradient(p -> loss_fn(p, 
							ls, states_list, controls_list, A, B), ps)[1]
	opt_state, ps = Optimisers.update(opt_state, ps, grad)
if epoch % 1000 == 1
	println("$(100 * round(epoch/N_EPOCHS; digits=3))%",
		" complete ... Loss is $(round(avg_loss, digits=5))")
	avg_loss = 0.0

	X, _  = Φₑ(TrainingData[:,1:end], ps.layer_1, ls.layer_1)
	#X⁺, _ = Φₑ(TrainingData[:,2:end], ps.layer_1, ls.layer_1)
	X⁺ = X[:,2:end]
	X = X[:,1:end-1]
	Γ = X⁺ / [X; U_rec[:,1:TrainingHorizon-1]]
	A = Γ[:,1:liftedDim]
	B = Γ[:,liftedDim+1:end]
	xTrunc0 = X[:,1]
	X, _ = Φd(X, ps.layer_2, ls.layer_2)

	PredFeaturesDeep = ls_lsim(A, B, 0, [U_rec 0], feature0=xTrunc0)
	PredFeaturesDeep, _ = Φd(PredFeaturesDeep, ps.layer_2, ls.layer_2)
	a1=plot(TrainingData[1,1:plotting_horizon])
	a1=plot!(X[1,1:plotting_horizon])
	a1 = plot!(PredFeaturesDeep[1,1:plotting_horizon])
	b1=plot(TrainingData[n+1,1:plotting_horizon])
	b1=plot!(X[n+1,1:plotting_horizon])
	b1 = plot!(PredFeaturesDeep[n+1,1:plotting_horizon])
	ab = plot(a1,b1, layout=(2,1))
	display(ab)
end
end
return ps, A, B
end

ps, Ann, Bnn = Train(params, opt_state, A, B)

println("Learning has been done!")
model = Φₑ
ps = ps.layer_1
ls = ls.layer_1
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