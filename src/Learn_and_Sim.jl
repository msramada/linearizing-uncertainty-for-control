# A supplementary code with functions used to generate simulation data

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
    

K_lqr = lqr(Discrete, A_true, B_true, I, I)

Qₛₛ = zero(System.A)
Qₛₛ[n+1:Nfeatures-1,n+1:Nfeatures-1] = I(Nfeatures-n-1)

Kₛₛ = lqr(System, Qₛₛ, I)
l₀ = make_info_state(x₀, Σ₀, dyna, infoType)
x_true1 = zeros(n, simHorizon+1)
x_DMD = zeros(NinfoState, simHorizon+1)
x_DMD[:,1] = l₀
x₁ = x₀
Σ₁ = Σ₀
U_DMD = zeros(1,simHorizon)
for k in 1:simHorizon
    u₀ = - Kₛₛ * make_feature(x_DMD[:,k])
    U_DMD[:,k] = u₀
    x_true1[:,k+1] = eKF.next_state_sample(x_true1[:,k], u₀, dyna)
    y_true = eKF.output_sample(x_true1[:,k], dyna)
    x₁, Σ₁ = eKF.update(x₁, Σ₁, u₀, y_true, dyna)
    x_DMD[:,k+1] = make_info_state(x₁, Σ₁, dyna, infoType)
end


x_true2 = zeros(n, simHorizon+1)
x_lqr = zeros(NinfoState, simHorizon+1)
x_lqr[:,1] = l₀
x₁ = x₀
Σ₁ = Σ₀
U_lqr = zeros(1,simHorizon)
for k in 1:simHorizon
    u₀ = - K_lqr * x_lqr[1:n,k]
    U_lqr[:,k] = u₀
    x_true2[:,k+1] = eKF.next_state_sample(x_true2[:,k], u₀, dyna)
    y_true = eKF.output_sample(x_true2[:,k], dyna)
    x₁, Σ₁ = eKF.update(x₁, Σ₁, u₀, y_true, dyna)
    x_lqr[:,k+1] = make_info_state(x₁, Σ₁, dyna, infoType)
end

return x_DMD, U_DMD, x_lqr, U_lqr, Qₛₛ, x_true1, x_true2
end

