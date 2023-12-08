X₀, X₁ = features[:,1:end-1], features[:,2:end]
ScalingVec = sum(X₀, dims = 2) ./ horizon
ScalingVec[1:NinfoState] .*= 1/2
#ScalingVec[1:n+1] ./= 2
S = inv(LinearAlgebra.Diagonal(ScalingVec[:]))
X₀, X₁ = S * X₀, S * X₁
Ω = [X₀; U_rec]

AB_mat = X₁ / Ω
A_mat = inv(S) * AB_mat[:,1:Nfeatures] * S
B_mat = inv(S) * AB_mat[:,Nfeatures+1:end]

System = ss(A_mat, B_mat, I, 0, 1)
pred_features,_ ,_ ,_ = lsim(System, U_rec, x0=features[:,1])

######### Plotting ###############
plotting_horizon = 300
kk = 1
p1 = plot(features[kk,1:plotting_horizon])
p1 = plot!(pred_features[kk,1:plotting_horizon])
#p1 = plot!(PredFeatures2[kk,1:plotting_horizon])
#p1 = plot!(PPredFeatures1[kk,1:plotting_horizon])

kk = n+1
p2 = plot(features[kk,1:plotting_horizon])
p2 = plot!(pred_features[kk,1:plotting_horizon])
#p2 = plot!(PredFeatures2[kk,1:plotting_horizon])
#p2 = plot!(PPredFeatures1[kk,1:plotting_horizon])


savefig("figs/pseudo.png")
display(plot(p1,p2, layout=(2,1), reuse = false))
