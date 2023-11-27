lₖ, U_rec

using Lux, Random, Optimisers

rng = Random.default_rng()
Random.seed!(rng, 0)
Nhidden = 30
Noutput = 25
model = Chain(Dense(NinfoState, Nhidden, relu),
	Dense(Nhidden, Nhidden, relu),
	Dense(Nhidden, Noutput))

ps, st = Lux.setup(rng, model)
