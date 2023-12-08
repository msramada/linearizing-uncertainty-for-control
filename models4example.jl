##### Dynamic System Definition #####
function observFun1(x)
	if (x > -0.0)
	return 0.1 .* x
	else
	return 0.5 .* x
	end
end

if systemNumber == 1
	function stateDynamics(x::Vector{Float64}, u::Vector{Float64})
		return 0.95 .* x + u
	end

	function outputDynamics(x::Vector{Float64})
		return observFun1.(x)
	end
	Q = LinearAlgebra.diagm([0.2])
	R = LinearAlgebra.diagm([0.1])
	Q_true = 0.2 * Q
	n = 1
	x₀ = randn(n,)
	Σ₀ = LinearAlgebra.diagm(rand(n,))

elseif systemNumber == 2
	function stateDynamics(x::Vector{Float64}, u::Vector{Float64})
		return 0.95 * x + u
	end

	function outputDynamics(x::Vector{Float64})
		return sqrt.(abs.(x)) .* sign.(x)
	end
	Q = LinearAlgebra.diagm([0.5])
	R = LinearAlgebra.diagm([0.5])
	Q_true = 0.2 * Q
	n = 1
	x₀ = randn(n,)
	Σ₀ = LinearAlgebra.diagm(rand(n,))

elseif systemNumber == 3
	function stateDynamics(x::Vector{Float64}, u::Vector{Float64})
		return [x[1] * x[2] + u[1]; x[2]]
	end

	function outputDynamics(x::Vector{Float64})
		return observFun1.([1 0] * x)
	end

	Q = LinearAlgebra.diagm([0.1, 0.05])
	R = LinearAlgebra.diagm([0.1])
	Q_true = LinearAlgebra.diagm([0.05, 0.0])
	n = 2
	x₀ = [randn(); 0.95]
	Σ₀ = LinearAlgebra.diagm(rand(n,))

elseif systemNumber == 4
	function stateDynamics(x::Vector{Float64}, u::Vector{Float64})
		v1 = 2 * sin(x[1]) + (5 + cos(x[1])) * u[1]
		v2 = -2 * cos(u[1]) * x[1] / (1 + x[1] ^ 2)
		return [v1; v2]
	end

	function outputDynamics(x::Vector{Float64})
		return observFun1.([1 0] * x)
	end

	Q = LinearAlgebra.diagm([0.3, 0.1])
	R = LinearAlgebra.diagm([0.2])
	Q_true = LinearAlgebra.diagm([0.1, 0.0])
	n = 2
	x₀ = [randn(); 0.95]
	Σ₀ = LinearAlgebra.diagm(rand(n,))

elseif systemNumber == 5
	function stateDynamics(x::Vector{Float64}, u::Vector{Float64})
		return [x[1] * x[2] + x[3] * u[1]; x[2]; x[3]]
	end

	function outputDynamics(x::Vector{Float64})
		return sqrt.(abs.([1 0 0] * x)) .* sign.([1 0 0] * x)
	end

	Q = LinearAlgebra.diagm([0.3, 0.1, 0.1])
	R = LinearAlgebra.diagm([0.2])
	Q_true = LinearAlgebra.diagm([0.1, 0.0, 0.0])
	n = 3
	x₀ = [randn(); 0.95; 1.0]
	Σ₀ = LinearAlgebra.diagm(rand(n,))
end
