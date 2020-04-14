function [G,A,iter] = sthosvd_adaptrng(X,order,tol,nr)
% Uses an adaptive version of the sequentially truncated HOSVD to compress
% a tensor X
%
% Inputs
% X - tensor to be compressed
% order - processing order of modes (vector)
% maxiter - max number of iterations for each mode
% tol - max allowable absolute error for entire tensor
% nr - number of new samples generated each iteration
%
% Outputs
% G - reduced order core tensor
% A - array of factor matrices
% iter - number of iterations taken for each mode (vector)
%
% - Written by Rachel Minster 2018

d = length(getsize(X));
sz = getsize(X);
m = length(order);

%initialize
G = X; 
A = cell(1,m);
iter = zeros(1,m);

for j = 1:m
    M = tens2mat(G, order(j));
    
    [Q,B,iter(j)] = randQB_EI_auto(M,tol/sqrt(d),nr,3);
    
    A{j} = Q;
    s = size(Q);
    sz(order(j)) = s(2);
    
    G = mat2tens(B,sz,order(j));
end
    
end