function [G,A] = rhosvd(X,r,p,maxiter,modes)
% Calculates randomized HOSVD of tensor X in Tucker form
% 
% Inputs
%   X: original tensor (d modes)
%   r: target rank vector [r1,r2,...,rd]
%   p: oversampling parameter (usually in the 5-20 range)
%   maxiter: max number of subspace iterations for randomized SVD
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
        Omega = randn(size(M,2),r(k)+p);
        [A{j},~,~] = randsvd(M,r(k),maxiter,Omega);         
    end
    
    G = tmprod(X,A,modes,'T');     %mode n product with transposes

end
