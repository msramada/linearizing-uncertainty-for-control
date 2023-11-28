using Lux, Random, Optimisers, Distributions, Statistics

η = 0.000001
N_EPOCHS = 10_000
N_LEAST_SQS = 5_0
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
X, _ = model(lₖ[:,1:end-1], params, ls)
X⁺, _ = model(lₖ[:,2:end], params, ls)
Γ = X⁺ / [X; U_rec]
A = Γ[:,1:liftedDim]
B = Γ[:,liftedDim+1:end]

function Train(A, B, ps, opt_state)
for j=1:N_LEAST_SQS
	avg_loss = 0.0
for epoch=1:N_EPOCHS
	indices = rand(1:horizon-1, BATCH_SIZE)
	πₖ = lₖ[:,indices]
	πₖ⁺ = lₖ[:,indices .+ 1]
	U = U_rec[:,indices]
	Loss = loss_fn(ps, ls, A, B, U, πₖ, πₖ⁺)
	avg_loss += Loss / N_EPOCHS
	grad  = Zygote.gradient(p -> loss_fn(p, ls, A, B, U, πₖ, πₖ⁺), ps)[1]
	#grad, _ = back((1.0, nothing))
	opt_state, ps = Optimisers.update(opt_state, ps, grad)
	push!(loss_history, Loss)
end
println("$(100 * round(j/N_LEAST_SQS; digits=3))% complete ... Loss is $(round(avg_loss, digits=2))")
if j<N_LEAST_SQS
	X, _ = model(lₖ[:,1:end-1], ps, ls)
	X⁺, _ = model(lₖ[:,2:end], ps, ls)
	Γ = X⁺ / [X; U_rec]
	A = Γ[:,1:liftedDim]
	B = Γ[:,liftedDim+1:end]
end
end
return A, B, ps
end

A, B, ps = Train(A, B, params, opt_state)

xTrunc0 = X[:,1]
PredFeaturesDeep = ls_lsim(A, B, 0, [U_rec 0], feature0=xTrunc0)
println(PredFeaturesDeep[1,10])
plotting_horizon = 300
start = 50
kk = 1
p1 = plot(features[kk,1:plotting_horizon])
p1 = plot!(PredFeaturesDeep[kk,1:plotting_horizon])

kk = n+1
p2 = plot(features[kk,1:plotting_horizon])
p2 = plot!(PredFeaturesDeep[kk,1:plotting_horizon])



plot(p1,p2, layout=(2,1), reuse = false)

#savefig("DeepKoop.png")
