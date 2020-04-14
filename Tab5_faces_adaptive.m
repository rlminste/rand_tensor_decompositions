%% Tab5_faces_adaptive.m
%
% This code generates Table 5 from the paper
%   'Randomized Algorithms for Low-rank Tensor
%    Decompositions in the Tucker Format'
%       - Minster, Saibaba, Kilmer
%
% Data from Olivetti Database of Faces, AT&T, 
%   https://cs.nyu.edu/~roweis/data.html

load olivettifaces.mat

X = zeros(40,4096,10);   %people x pixels x poses
for i = 1:40
    X(i,:,:) = faces(:,1+10*(i-1):10*i);
end

Xnrm = frob(X);
modes = [2,1,3];
d = length(size(X));


% adaptive R-STHOSVD
tol = [0.25,0.2,0.15,0.1,0.05,0.01];
l = length(tol);

sz = ones(l,d);
adapterr = zeros(1,l);
st_err = zeros(1,l);

% Adaptive R-STHOSVD
for j = 1:l
    [G,A,iter,~] = sthosvd_adaptrng(X,modes,tol(j),1);
    sz(j,1:d) = size(G,1:d);
    
    T = tmprod(G,A,modes);
    adapterr(j) = frob(X-T)/Xnrm;
end

% STHOSVD
for j = 1:l
    [G,A] = sthosvd(X,sz(j,:),modes);
    
    T = tmprod(G,A,modes);
    st_err(j) = frob(X-T)/Xnrm;
end

disp('Rank r of adaptive STHOSVD approximation (rows are ranks)')
disp(sz)

disp('Actual error of adaptive approximation')
disp(adapterr)

disp('STHOSVD error with rank r')
disp(st_err)