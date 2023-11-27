AllConstants = X₁ / Ω 
BG_polys = X₁ / Ω₂
G_poly = X₁ / Ω₃
A_mat = inv(S) * AllConstants[:,1:Nfeatures] * S
B_mat = inv(S) * reshape(AllConstants[:,Nfeatures+1], (Nfeatures,1))
G_mat = inv(S) * reshape(AllConstants[:,Nfeatures+2], (Nfeatures,1))

A_mat1 = inv(S) * BG_polys[:,1:Nfeatures] * S
B_mat1 = inv(S) * reshape(BG_polys[:,Nfeatures+1], (Nfeatures,1)) .* 0
G_mat1 = inv(S) * reshape(BG_polys[:,Nfeatures+2], (Nfeatures,1)) .* 0
Bx_mat1 = inv(S) * BG_polys[:,Nfeatures+1:Nfeatures+1+Nfeatures-1] * S 
Gx_mat1 = inv(S) * BG_polys[:,Nfeatures+1+Nfeatures:end] * S

A_mat2 = inv(S) * G_poly[:,1:Nfeatures] * S
B_mat2 = inv(S) * reshape(G_poly[:,Nfeatures+1], (Nfeatures,1))
Gx_mat2 = inv(S) * G_poly[:,Nfeatures+2:end] * S 

sys_uy = ss(A_mat, [B_mat G_mat], I, 0, 1)
pred_features,_ ,_ ,_ = lsim(sys_uy,[U_rec 0; Y_rec[:,2:end] 0], x0=features[:,1])

feature0 = features[:,1]
pred_features2 = ls_lsim(A_mat1, B_mat1, G_mat1, Bx_mat1, Gx_mat1, U_rec , Y_rec, feature0)
pred_features3 = ls_lsim(A_mat2, B_mat2, 0 .* G_mat1, 0 .* Bx_mat1, Gx_mat2, U_rec , Y_rec, feature0)


kk=1; p1 = plot(features[kk,1:plotting_horizon], lw=2); 
p1 = plot!(pred_features[kk,1:plotting_horizon],lw=1); 
p1 = plot!(pred_features2[kk,1:plotting_horizon],lw=1); 
#p1 = plot!(pred_features3[kk,1:plotting_horizon],lw=1); 
p1 = plot!(x_true[1,1:plotting_horizon])

kk=n+1; p2 = plot(features[kk,1:plotting_horizon], lw=2); 
p2 = plot!(pred_features[kk,1:plotting_horizon],lw=1)
p2 = plot!(pred_features2[kk,1:plotting_horizon],lw=1)
p2 = plot!(pred_features3[kk,1:plotting_horizon],lw=1)

error(x,y) = sum((x .- y).^2)
kk=4
println("With constants: ", error(features[kk,:], pred_features[kk,:]))
println("Control and Output affine: ", error(features[kk,:], pred_features2[kk,:]))
println("Control constant and Output affine: ", error(features[kk,:], pred_features3[kk,:]))


plot(p1,p2, layout=(2,1))
