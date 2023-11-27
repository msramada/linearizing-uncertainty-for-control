module featureVecMaker


function relu(x)
	if x>0
		return x
	else
		return 0.01 * x
	end
end



int(x) = floor(Int, x)
number = 15
n = 2
BIAS = randn(int(n + (n^2+n)/2 * 3), number-1)
SIGN = sign.(randn(size(BIAS))) .* 1.0

function makeFeature_polys(l::Vector{Float64}) # returns a vector of length # * (nn²+nn)/2 + nn
	l = [[1.0];l]
	l₁= vectorize(l * l')
	l₂= vectorize(l₁ * l') 
	l₃= vectorize(l₂ * l')
#	l₄= vectorize(l₃ * l')
	l₄ = l₃
	l₄= ([l₄;l₄[1]])[2:end]
	lₓ= 1 ./ (l[1] ^ 4 + 0.01) * l[2]
	lᵥ= 1 ./ (l[1] ^ 4 + 0.01)
	return [l₄; lₓ; lᵥ] # n + (n²+n)/2 * 3
end

function makeFeature_1(l::Vector{Float64}) # returns a vector of length # * (nn²+nn)/2 + nn
	l = [[1.0];l]
	l₁ = vectorize(l * l')
	return l₁ # n + (n²+n)/2 * 3
end

function makeFeature_relus(l::Vector{Float64}) # returns a vector of length 2n²+3n
	return reluFeatures(makeFeature_1(l))
end


function reluFeatures(b::Vector{Float64}) # 
	m = length(b)
	B = b
	for i in 1:number-1
		B = [B; relus.(SIGN[:,i] .* b + BIAS[:,i])]
	end
	return B
end

end
