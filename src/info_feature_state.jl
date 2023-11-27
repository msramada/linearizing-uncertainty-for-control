int(x) = floor(Int, x)

function make_info_state(x::Vector{Float64}, Σ::Matrix{Float64}, DynamicSysObj, infoType::String)
	if infoType == "trace"
		return [x; sqrt.(LinearAlgebra.tr(Σ))]
	elseif infoType == "cholesky"
		n = size(DynamicSysObj.Q)[1]
		l = zeros(int((n^2+n)/2),)
		Σ = Symmetric(Σ)# + 0.1 * I)
		L = cholesky(Σ).L
		#L = Σ
		for i in 1:n
			l[int((i^2-i)/2+1):int((i^2+i)/2)] = L[i,1:i]
		end
		#l = log.(l)
		return [x;l]
	end
end

function back_from_cholesky_Σ(l::Vector{Float64}, DynamicSysObj)
	n = size(DynamicSysObj.Q)[1]
	L = zeros(n, n)
	x = l[1:n]
	#x = exp.(x)
	vect = l[n+1:end]
	for i in 1:n
		L[i,1:i] = vect[int((i^2-i)/2+1):int((i^2+i)/2)]
	end
	Σ = L * L'
	#Σ = (L + L') .- LinearAlgebra.diagm(LinearAlgebra.diag(L))
	return x, Σ
end

function vector_infoState_update(π₀::Vector{Float64}, u₀::Vector{Float64}, y₁::Vector{Float64}, DynamicSysObj)
	n = size(DynamicSysObj.Q)[1]
	x₀₀, Σ₀₀ = to_matvec_infoState(π₀, DynamicSysObj)
	x₁₁, Σ₁₁ = eKF.update(x₀₀, Σ₀₀, u₀, y₁, DynamicSysObj)
	return to_vector_infoState(x₁₁, Σ₁₁, DynamicSysObj)
end


function vectorize(A::Matrix)
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

function makeFeature1(l::Vector{Float64}, order::Int, DynamicSysObj)
	return [makeFeature_Polys(l, order, DynamicSysObj);
		makeFeature_Obs(l, order, DynamicSysObj); 1.0]
end
function makeFeature_Polys(l::Vector{Float64}, order::Int, DynamicSysObj) # returns a vector of length # * (nn²+nn)/2 + nn
	lₚ = l
	for i in 1:order
		lₚ = vectorize(lₚ * l')
	end
	return [l; lₚ; 1.0]
end

function makeFeature_Obs(l::Vector{Float64}, order::Int, DynamicSysObj) # returns a vector of length # * (nn²+nn)/2 + nn
	g = DynamicSysObj.h(l[1:DynamicSysObj.n])
	g = [1.0; g]
	lₚ = g
	for i in 1:order+6
		lₚ = vectorize(lₚ * g')
	end
	return lₚ
end
