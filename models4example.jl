##### Dynamic System Definition #####
function observFun1(x)
	if (x > -1)
	return 0.2 .* x
	else
	return 0.5 .* x
	end
end

if systemNumber == 1
	function stateDynamics(x::Vector{Float64}, u::Vector{Float64})
		return 0.95 * x + u
	end

	function outputDynamics(x::Vector{Float64})
		return observFun1.(x)
	end
	Q = LinearAlgebra.diagm([1])
	R = LinearAlgebra.diagm([1])
	Q_true = 0.2 * Q
	n = 1
	x₀ = randn(n,)
	Σ₀ = LinearAlgebra.diagm(rand(n,))

elseif systemNumber == 2
	function stateDynamics(x::Vector{Float64}, u::Vector{Float64})
		return 0.95 * x + u
	end

	function outputDynamics(x::Vector{Float64})
		return 1/27 .* x .^ 3
	end
	Q = LinearAlgebra.diagm([1])
	R = LinearAlgebra.diagm([1])
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

	Q = LinearAlgebra.diagm([0.2, 0.1])
	R = LinearAlgebra.diagm([0.1])
	Q_true = LinearAlgebra.diagm([0.1, 0.0])
	n = 2
	x₀ = [randn(); 0.95]
	Σ₀ = LinearAlgebra.diagm(rand(n,))
end
