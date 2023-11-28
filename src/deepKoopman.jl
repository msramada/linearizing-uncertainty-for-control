using Lux, Random, Optimisers, Distributions, Statistics

N_SAMPLES = 200
η = 0.1
N_EPOCHS = 1_00
N_LEAST_SQS = 1_00
rng = Random.default_rng()
Random.seed!(rng, 0)
Nhidden = 30
Noutput = liftedDim
BATCH_SIZE = 64
model = Chain(
	Dense(NinfoState, Nhidden, relu),
	Dense(Nhidden, Nhidden, relu),
	Dense(Nhidden, Noutput)
	)

params, ls = Lux.setup(rng, model)

function loss_fn(p, ls, A, B, U, X, X⁺)
	Features, _ = model(X, p, ls)
	Features_pred, _ = model(X⁺, p, ls)
	Features⁺ = A * Features + B * U
	loss = 0.5 * mean((Features⁺ .- Features_pred) .^ 2)
	return loss
end

opt = Optimisers.Adam(η)
opt_state = Optimisers.setup(opt, params)

A = randn(liftedDim, liftedDim)
B = randn(liftedDim, 1)
loss_history = []


for j in 1:N_LEAST_SQS
for epoch in 1:N_EPOCHS
	indices = rand(1:horizon-1, BATCH_SIZE)
	πₖ = lₖ[:,indices]
	πₖ⁺ = lₖ[:,indices .+ 1]
	U = U_rec[:,indices]
	global ps = params
	Loss, _ = Zygote.pullback(p -> loss_fn(p, ls, A, B, U, πₖ, πₖ⁺), ps)
	grad, _ = back((1.0, nothing))
	ops, ps = Optimisers.update(opt_state, params, grad)
	params = ps
	opt_state = ops
	push!(loss_history, Loss)
	if epoch % 100 == 0
		println("Epoch: $Epoch")
end
end
X, _ = model(lₖ[1:end-1], p, ls)
X⁺, _ = model(lₖ[2:end], p, ls)
Γ = X⁺ / [X; U_rec]
A = Γ[1:NinfoState,:]
B = Γ[NinfoState+1:end,:]
end