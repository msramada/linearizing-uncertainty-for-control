using Lux, Random, Optimisers, Distributions, Statistics

η = 0.000001
N_EPOCHS = 1_000
N_LEAST_SQS = 1_00
rng = Random.default_rng()
Random.seed!(rng, 0)
Nhidden = 64
Noutput = liftedDim
BATCH_SIZE = 32
model = Chain(
	Dense(NinfoState, Nhidden, relu),
	Dense(Nhidden, Nhidden, relu),
	Dense(Nhidden, Nhidden, relu),
	Dense(Nhidden, Noutput)
	)

params, ls = Lux.setup(rng, model)
TrainingHorizon = floor(Int, size(lₖ)[2]/2)
TrainingData = lₖ[:,1:TrainingHorizon]
ValidationData = lₖ[:,TrainingHorizon+1:end]

function loss_fn(p, ls, A, B, U, X, X⁺)
	Features, _ = model(X, p, ls)
	Features_pred, _ = model(X⁺, p, ls)
	Features⁺ = A * Features + B * U
	loss1 = 1 * mean((Features⁺ .- Features_pred) .^ 2)
	loss2 = 1 * mean((Features[1:NinfoState,:] .- X) .^ 2)
	return loss1 + loss2
end

opt = Optimisers.Adam(η)
opt_state = Optimisers.setup(opt, params)

A = randn(liftedDim, liftedDim)
B = randn(liftedDim, 1)
loss_history = []

X, _ = model(TrainingData[:,1:end-1], params, ls)
X⁺, _ = model(TrainingData[:,2:end], params, ls)
Γ = X⁺ / [X; U_rec[:,1:TrainingHorizon-1]]
A = Γ[:,1:liftedDim]
B = Γ[:,liftedDim+1:end]
plotting_horizon = 200
function Train(A, B, ps, opt_state)
for j=1:N_LEAST_SQS
	avg_loss = 0.0
for epoch=1:N_EPOCHS
	indices = rand(1:TrainingHorizon - 1, BATCH_SIZE)
	πₖ = TrainingData[:,indices]
	πₖ⁺ = TrainingData[:,indices .+ 1]
	U = U_rec[:,indices]
	Loss = loss_fn(ps, ls, A, B, U, πₖ, πₖ⁺)
	avg_loss += Loss / N_EPOCHS
	grad  = Zygote.gradient(p -> loss_fn(p, ls, A, B, U, πₖ, πₖ⁺), ps)[1]
	#grad, _ = back((1.0, nothing))
	opt_state, ps = Optimisers.update(opt_state, ps, grad)
	push!(loss_history, Loss)
end
println("$(100 * round(j/N_LEAST_SQS; digits=3))% complete ... Loss is $(round(avg_loss, digits=2))")
if j<=N_LEAST_SQS
	X, _ = model(TrainingData[:,1:end-1], ps, ls)
	X⁺, _ = model(TrainingData[:,2:end], ps, ls)
	Γ = X⁺ / [X; U_rec[:,1:TrainingHorizon-1]]
	A = Γ[:,1:liftedDim]
	B = Γ[:,liftedDim+1:end]
	xTrunc0 = X[:,1]
	PredFeaturesDeep = ls_lsim(A, B, 0, [U_rec 0], feature0=xTrunc0)
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
return A, B, ps
end
Ann, Bnn, ps = Train(A, B, params, opt_state)
println("done")
X, _ = model(ValidationData[:,1:end-1], ps, ls)
a1=plot(ValidationData[1,1:plotting_horizon])
a1=plot!(X[1,1:plotting_horizon])
b1=plot(ValidationData[n+1,1:plotting_horizon])
b1=plot!(X[n+1,1:plotting_horizon])
ab = plot(a1,b1, layout=(2,1))
display(ab)

#=
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