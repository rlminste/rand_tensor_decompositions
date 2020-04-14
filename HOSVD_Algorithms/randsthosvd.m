function [G,A] = randsthosvd(X,r,order,p,q)
% Calculates randomized STHOSVD of tensor X in Tucker form
% 
% Inputs
%   X: original tensor (d modes)
%   r: target rank vector [r1,r2,...,rd]
%   order: processing order vector of modes (can be less than length d)
%   p: oversampling paramter (usually in the 5-20 range)
%   q: max number of subspace iterations for randomized SVD
%
% Outputs
%   G: core tensor (size(G) = r)
%   A: cell array of factor matrices
%
% Written by Rachel Minster, 2018

G = X; 
sz = getsize(X);
m = length(order);
A = cell(1,m);

for j = 1:m
    k = order(j);
    M = tens2mat(G,k); %matricize along mode k
    
    Omega = randn(size(M,2),r(k)+p);
    [Q,~,~] = randsvd(M,r(k),q,Omega);  %randomized SVD
    
    A{j} = Q;
    sz(k) = r(k);
    
    G = mat2tens(Q'*M,sz,k); %reform core tensor
end
end