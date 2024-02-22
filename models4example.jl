using Lux
##### Dynamic System Definition #####

A_true = [0.63 0.54 0.0; 0.74 0.96 0.68; 0.1 -0.86 0.54]
B_true = [0.0;1.0;0.0;;]
function stateDynamics(x::Vector{Float64}, u::Vector{Float64})
	return A_true * x + B_true * u
end

function outputDynamics(x::Vector{Float64})
	return [elu.( 1.0/1.0 * (1.0 .*(x[1] + x[2] + x[3]) .- 3.0))]
end
σ = 0.2
Q = LinearAlgebra.diagm([σ,σ,σ])
R = LinearAlgebra.diagm([σ])
Q_true = 1.0 * Q
n = 3
x₀ = randn(n,)
Σ₀ = LinearAlgebra.diagm(rand(n,))

