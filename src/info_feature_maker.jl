module info_feature_maker
using LinearAlgebra
export make_info_state, back_from_cholesky_Σ, vector_infoState_update, vectorize_all
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

end