int(x) = floor(Int, x)

function make_info_state(x::Vector{Float64}, Σ::Matrix{Float64}, DynamicSysObj, infoType::String)
	if infoType == "trace"
		return [x; sqrt.(LinearAlgebra.tr(Σ))]
	elseif infoType == "cholesky"
		n = size(DynamicSysObj.Q)[1]
		l = zeros(int((n^2+n)/2),)
		Σ = Symmetric(Σ)
		L = cholesky(Σ).L
		l = 1.0 * half_vectorize(Matrix(L))
		return [x;l]
	end
end

function back_from_cholesky_Σ(l::Vector{Float64}, DynamicSysObj)
	n = size(DynamicSysObj.Q)[1]
	L = zeros(n, n)
	x = l[1:n]
	vect = l[n+1:end]
	counter = 0
	for j in 1:n
		for i in 1:n
			if j<=i
				counter += 1
				L[i,j] = vect[counter]
			end
		end
	end
	Σ = L * L'
	return x, Σ
end

function vector_infoState_update(π₀::Vector{Float64}, u₀::Vector{Float64}, y₁::Vector{Float64}, DynamicSysObj)
	n = size(DynamicSysObj.Q)[1]
	x₀₀, Σ₀₀ = to_matvec_infoState(π₀, DynamicSysObj)
	x₁₁, Σ₁₁ = eKF.update(x₀₀, Σ₀₀, u₀, y₁, DynamicSysObj)
	return to_vector_infoState(x₁₁, Σ₁₁, DynamicSysObj)
end


function half_vectorize(A::Matrix)
	n, m = size(A)
	l = []
	for j in 1:m
		for i in 1:n
			if j<=i
				l = [l; A[i,j]]
			end
		end
	end
	return l
end

function vectorize_all(A::Matrix)
	n, m = size(A)
	l = []
	for j in 1:m
		for i in 1:n
				l = [l; A[i,j]]
		end
	end
	return l
end

function makeFeature1(l::Vector{Float64}, order::Int, DynamicSysObj)
	return [makeFeature_Polys(l, order, DynamicSysObj);
		makeFeature_Obs(l, order, DynamicSysObj)]
end
function makeFeature_Polys(l::Vector{Float64}, order::Int, DynamicSysObj) # returns a vector of length # * (nn²+nn)/2 + nn
	l = [1.0; l]
	lₚ = l
	for i in 1:order
		lₚ = half_vectorize(lₚ * l')
	end
	lₚ = [lₚ[2:end];lₚ[1]]
	return lₚ
end

function makeFeature_Trigon(l::Vector{Float64}, order::Int, DynamicSysObj) # returns a vector of length # * (nn²+nn)/2 + nn
	ω = 0.1:0.1:2
	lₚ = l
	for i in 1:length(ω)
		lₚ = [lₚ; sin.(ω[i] .* l); cos.(ω[i] .* l)]
	end
	lₚ = [lₚ; 1.0]
	return lₚ
end

function makeFeature_Obs(l::Vector{Float64}, order::Int, DynamicSysObj) # returns a vector of length # * (nn²+nn)/2 + nn
	g = DynamicSysObj.h(l[1:DynamicSysObj.n])
	g = [1.0; g]
	lₚ = g
	for i in 1:order+6
		lₚ = half_vectorize(lₚ * g')
	end
	return lₚ[2:end]
end
