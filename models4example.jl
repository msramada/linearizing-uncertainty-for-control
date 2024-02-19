using Lux
##### Dynamic System Definition #####

A_true =  [0.56 0.44 0.0; 0.44 1.06 0.48; 0.0 -0.66 0.48]
B_true = [0.0;1.0;0.0;;]
function stateDynamics(x::Vector{Float64}, u::Vector{Float64})
	return A_true * x + B_true * u
end

function outputDynamics(x::Vector{Float64})
	return [elu.( 1.0/1.0 * (1.0 .*(x[1] + x[2] + x[3]) .- 3.0))]
end
Q = LinearAlgebra.diagm([0.3,0.3,0.3])
R = LinearAlgebra.diagm([0.3])
Q_true = 1.0 * Q
n = 3
x₀ = randn(n,)
Σ₀ = LinearAlgebra.diagm(rand(n,))

