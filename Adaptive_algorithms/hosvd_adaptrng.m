function [G,A,iter,err] = hosvd_adaptrng(X,tol,nr,modes)
% Uses an adaptive version of the HOSVD to compress a tensor
%
% Inputs
% X - tensor to be compressed
% maxiter - max number of iterations for each mode
% tol - largest absolute error allowable for entire tensor
% nr - number of new samples generated each iteration
%
% Outputs
% G - reduced order core tensor
% A - array of factor matrices
% iter - number of iterations used for each mode (vector)
% err - actual error of adaptive matrix approximation
%
% Written by Rachel Minster, 2018

d = length(getsize(X));
m = length(modes);
iter = zeros(1,m);
err = zeros(1,m);
A = cell(1,m);

for i = 1:m
    j = modes(i);
    M = tens2mat(X,j);
    [A{i},~,iter(j),err(j)] = randQB_EI_auto(M,tol/sqrt(d),nr,2);
    
end

% form core tensor
G = tmprod(X,A,modes,'T');
end
