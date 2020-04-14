function [G,A] = hosvd(X,r,modes)
% Calculates HOSVD of tensor X in Tucker form
% 
% Inputs
%   X: original tensor
%   r: target rank vector [r1,r2,...,rd]
%   modes: vector of modes to compress (order doesn't matter)
%
% Outputs
%   G: core tensor (size(G) = r)
%   A: cell array of factor matrices
%
% Written by Rachel Minster, 2018

m = length(modes);
A = cell(1,m);

for j = 1:m
    k = modes(j);
    M = tens2mat(X,k);      %matricize along mode j
   [U,~,~] = svd(M,'econ');  
    A{j} = U(:,1:r(k));     %save first rj left s-vectors as factor matrix
end

G = tmprod(X,A,modes,'T');     %mode n product with transposes

end
