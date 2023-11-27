X₀, X₁ = features[:,1:end-1], features[:,2:end]
ScalingVec = sum(X₀, dims = 2) ./ horizon
ScalingVec[1:NinfoState] .*= 1/2
#ScalingVec[1:n+1] ./= 2
S = inv(LinearAlgebra.Diagonal(ScalingVec[:]))
X₀, X₁ = S * X₀, S * X₁

XX = features[:,delays+1:end]
for i in 1:delays
	global XX = [XX;features[:,delays+1-i:end-i]]
end
XX₀ = XX[:,1:end-1]
XX₁ = XX[:,2:end]
ScalingVec = sum(XX₀, dims = 2) ./ horizon
SS = inv(LinearAlgebra.Diagonal(ScalingVec[:])) #LinearAlgebra.Diagonal(1 ./ ScalingVec[:])
XX₀ = SS * XX₀
XX₁ = SS * XX₁

Ω = [X₀; U_rec]
Ω₂ = [X₀; U_rec; X₀ .* U_rec]
ΩΩ = [XX₀; U_rec[:,delays+1:end]]
numberOfstates = 20
reduction = 1 - numberOfstates / Nfeatures
Truncation = floor(Int, Nfeatures * (1-reduction))
Truncation1 = floor(Int, 2 * Truncation)#Truncation + floor(Int, reduction * Truncation)
Truncation2 = 2 * Truncation
function trunc_svd(Mat, trunc)
	U, s, V = svd(Mat)
	U = U[:,1:trunc]
	S = Diagonal(s[1:trunc])
	V = V[:,1:trunc]
	return U, S, V
end

######### Hankel here ##########
Ũ, S̃, Ṽ = trunc_svd(ΩΩ, (delays+1) * Truncation1)
Û, Ŝ, V̂ = trunc_svd(XX₁, (delays+1) * Truncation)

Ũ₁ = Ũ[1:(delays+1)*Nfeatures,:]
Ũ₂ = Ũ[(delays+1)*Nfeatures+1:end,:]

Ã = Û' * XX₁ * Ṽ * inv(S̃) * Ũ₁' * Û
B̃ = Û' * XX₁ * Ṽ * inv(S̃) * Ũ₂'

xTrunc0 = Û' * SS * XX[:,1]

SSysTruncated1 = ss(Ã, B̃, I, 0, 1)
TTruncatedFeatures1,_ ,_ ,_ = lsim(SSysTruncated1,[U_rec[:,delays+1:end-1] 0], 
			x0=xTrunc0)
PPredFeatures1 = inv(SS) * Û * TTruncatedFeatures1
PPredFeatures1[:,1] = XX[:,1]

######### All Constants ##########
Ũ, S̃, Ṽ = trunc_svd(Ω, Truncation1)
Û, Ŝ, V̂ = trunc_svd(X₁, Truncation)

Ũ₁ = Ũ[1:Nfeatures,:]
Ũ₂ = Ũ[Nfeatures+1:end,:]

Ã = Û' * X₁ * Ṽ * inv(S̃) * Ũ₁' * Û 
B̃ = Û' * X₁ * Ṽ * inv(S̃) * Ũ₂'

xTrunc0 = Û' * S * features[:,1]
#K₁ * Û' * S * features[:,1]
SysTruncated1 = ss(Ã, B̃, I, 0, 1)
TruncatedFeatures1,_ ,_ ,_ = lsim(SysTruncated1,[0 U_rec], x0=xTrunc0)
PredFeatures1 = inv(S) * Û * TruncatedFeatures1
PredFeatures1 = [PredFeatures1[:,2:end] zero.(PredFeatures1[:,1])]
PredFeatures1[:,1] = features[:,1]
#K₁ = lqr(SysTruncated1, I, I)

######### Carlemann ###############
Ū, S̄, V̄ = trunc_svd(Ω₂, Truncation2)

Ū₁ = Ū[1:Nfeatures,:]
Ū₂ = Ū[Nfeatures+1:Nfeatures+1,:]
Ū₃ = Ū[Nfeatures+2:end,:]

Ā = Û' * X₁ * V̄ * inv(S̄) * Ū₁' * Û 
B̄ = Û' * X₁ * V̄ * inv(S̄) * Ū₂'
B̄ₓ = Û' * X₁ * V̄ * inv(S̄) * Ū₃'

function ls_lsim(A, B, Bx, U_rec; feature0)
horizon = size(U_rec)[2]
pred_features = zeros(size(A)[1], horizon+1)
pred_features[:,1] = feature0
U_rec1=U_rec[1:1,:]
for k in 1:horizon
	pred_features[:,k+1] = (A * pred_features[:,k]
		+ B * U_rec[:,k] 
		+ Bx * pred_features[:,k] .* U_rec1[:,k])
end
	return pred_features
end

PredFeatures2 = ls_lsim(Ā, B̄, B̄ₓ * Û, [U_rec 0], feature0=xTrunc0)
PredFeatures2 = inv(S) * Û * PredFeatures2
PredFeatures2[:,1] = features[:,1]

######### Plotting ###############
plotting_horizon = 300
start = 50
kk = 1
p1 = plot(features[kk,1:plotting_horizon])
p1 = plot!(PredFeatures1[kk,1:plotting_horizon])
p1 = plot!(PredFeatures2[kk,1:plotting_horizon])
#p1 = plot!(PPredFeatures1[kk,1:plotting_horizon])

kk = n+1
p2 = plot(features[kk,1:plotting_horizon])
p2 = plot!(PredFeatures1[kk,1:plotting_horizon])
p2 = plot!(PredFeatures2[kk,1:plotting_horizon])
#p2 = plot!(PPredFeatures1[kk,1:plotting_horizon])



display(plot(p1,p2, layout=(2,1), reuse = false))
#savefig("myplot.png")