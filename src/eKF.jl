module eKF
using Zygote, LinearAlgebra

struct StateSpaceSys
	f::Function
	h::Function
	Q::Matrix
	R::Matrix
	Q_true::Matrix
	n::Int
	m::Int

	function StateSpaceSys(f, h, Q, R, Q_true)
		new(f, h, Q, R, Q_true, size(Q)[1], size(R)[1])
	end
	
end


function ∇f(x₀::Vector{Float64}, u₀::Vector{Float64}, DynamicSysObj)
	Zygote.jacobian(x -> DynamicSysObj.f(x, u₀), x₀)[1]
end

function ∇h(x₀::Vector{Float64}, DynamicSysObj)
	Zygote.jacobian(x -> DynamicSysObj.h(x), x₀)[1]
end

function time_update(x₀₀::Vector{Float64}, Σ₀₀::Matrix{Float64}, u₀::Vector{Float64}, DynamicSysObj)
	x₁₀ = DynamicSysObj.f(x₀₀,u₀)
	F = ∇f(x₀₀, u₀, DynamicSysObj) # jacobian of state dynamics
	Σ₁₀ = F * Σ₀₀ * F' + DynamicSysObj.Q
	return x₁₀, Σ₁₀
end

function measurement_update(x₁₀::Vector{Float64}, Σ₁₀::Matrix{Float64}, y₁::Vector{Float64}, DynamicSysObj)
	H = ∇h(x₁₀, DynamicSysObj) # jacobian of measurement dynamics
	L = Σ₁₀ * H' * inv(H * Σ₁₀ * H' + DynamicSysObj.R) #Kalman Gain
	x₁₁ = x₁₀ + L * (y₁ - DynamicSysObj.h(x₁₀))
	Σ₁₁ = (I - L * H) * Σ₁₀
	return x₁₁, Σ₁₁
end

function update(x₀₀::Vector{Float64}, Σ₀₀::Matrix{Float64}, u₀, y₁::Vector{Float64}, DynamicSysObj)
	x₁₀, Σ₁₀ = time_update(x₀₀, Σ₀₀, u₀, DynamicSysObj)
	x₁₁, Σ₁₁ = measurement_update(x₁₀, Σ₁₀, y₁, DynamicSysObj)
	return x₁₁, Σ₁₁
end

function update_predictor(x₁₀::Vector{Float64}, Σ₁₀::Matrix{Float64}, u₁, y₁::Vector{Float64}, DynamicSysObj)
	x₁₁, Σ₁₁ = measurement_update(x₁₀, Σ₁₀, y₁, DynamicSysObj)
	x₂₁, Σ₂₁ = time_update(x₁₁, Σ₁₁, u₁, DynamicSysObj)
	return x₂₁, Σ₂₁
end

function next_state_sample(x₀::Vector{Float64}, u₀::Vector{Float64}, DynamicSysObj)
	x₁ = DynamicSysObj.f(x₀, u₀) + sqrt(DynamicSysObj.Q_true) * randn(DynamicSysObj.n,)
	return x₁
end

function output_sample(x₁::Vector{Float64}, DynamicSysObj)
	ȳ = DynamicSysObj.h(x₁)
	y₁ = ȳ + sqrt(DynamicSysObj.R) * randn(DynamicSysObj.m,)
	return y₁
end


end