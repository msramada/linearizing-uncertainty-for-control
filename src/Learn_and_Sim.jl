function learn_system(features::Matrix, U_rec::Matrix)

X₀, X₁ = features[:,1:end-1], features[:,2:end]
horizon = size(features)[2]
ScalingVec = sum(X₀, dims = 2) ./ horizon
ScalingVec[1:NinfoState] .*= 1/2
S = inv(LinearAlgebra.Diagonal(ScalingVec[:]))
X₀, X₁ = S * X₀, S * X₁
Ω = [X₀; U_rec]

AB_mat = X₁ / Ω
A_mat = inv(S) * AB_mat[:,1:Nfeatures] * S
B_mat = inv(S) * AB_mat[:,Nfeatures+1:end]

System = ss(A_mat, B_mat, I, 0, 1)
learnt_features,_ ,_ ,_ = lsim(System, U_rec, x0=features[:,1])

return learnt_features, System
end

##################

function simulate_system(System, simHorizon::Int, x₀, Σ₀)
    
A = [α;;]
B = [β;;]
K_lqr = lqr(Discrete, A, B, I, I)

Qₛₛ = zero(System.A)
Qₛₛ[n+1:n+NinfoState,n+1:n+NinfoState] = I(NinfoState)

Kₛₛ = lqr(System, Qₛₛ, I)
l₀ = make_info_state([x₀;zero(x₀)], Σ₀, dyna, infoType)
x_true1 = zeros(n, simHorizon+1)
x_DMD = zeros(NinfoState, simHorizon+1)
x_DMD[:,1] = l₀
x₁ = x₀
Σ₁ = Σ₀
for k in 1:simHorizon
    u₀ = - Kₛₛ * make_feature(x_DMD[:,k])
    x_true1[:,k+1] = eKF.next_state_sample(x_true1[:,k], u₀, dyna)
    y_true = eKF.output_sample(x_true1[:,k+1], dyna)
    Σ₀ = Σ₁
    x₀ = x₁
    x₁, Σ₁ = eKF.update(x₁, Σ₁, u₀, y_true, dyna)
    x_DMD[:,k+1] = make_info_state([x₁;x₀], Σ₀, dyna, infoType)
end


x_true2 = zeros(n, simHorizon+1)
x_lqr = zeros(NinfoState, simHorizon+1)
x_lqr[:,1] = l₀
x₁ = x₀
Σ₁ = Σ₀
for k in 1:simHorizon
    u₀ = - K_lqr * x_lqr[1:1,k]
    #U_rec[:,k] = u₀
    x_true2[:,k+1] = eKF.next_state_sample(x_true2[:,k], u₀, dyna)
    y_true = eKF.output_sample(x_true2[:,k+1], dyna)
    Σ₀ = Σ₁
    x₀ = x₁
    x₁, Σ₁ = eKF.update(x₁, Σ₁, u₀, y_true, dyna)
    x_lqr[:,k+1] = make_info_state([x₁;x₀], Σ₀, dyna, infoType)
end

return x_DMD, x_lqr, Qₛₛ, x_true1, x_true2
end