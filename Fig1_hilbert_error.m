%% Fig1_hilbert_error.m
%
% This code generates Figure 1 from the paper
%   'Randomized Algorithms for Low-rank Tensor
%    Decompositions in the Tucker Format'
%       - Minster, Saibaba, Kilmer

%% define Hilbert tensor
n = 25;
d = [n,n,n,n,n];
[ii,ij,ik,il,im] = ndgrid(1:d(1),1:d(2),1:d(3),1:d(4),1:d(5));
X = 1./(ii+ij+ik+il+im);

%% variables
p = 5;   % oversampling parameter
maxiter = 0; % number of subspace iterations
mode = length(getsize(X)); % number of modes of the tensor
order = [1,2,3,4,5];

%% initializing
h_err = zeros(1,n);
r_err = zeros(1,n);
st_err = zeros(1,n);
rst_err = zeros(1,n);
bound = zeros(n,1);

%% HOSVD
for r = 1:n
    [G,A] = hosvd(X,[r,r,r,r,r],1:mode);
    T = tmprod(G,A,1:mode);

    h_err(r) = frob(X-T)/frob(X);
end

%% RHOSVD
for r = 1:n
    [G,A] = rhosvd(X,[r,r,r],p,maxiter,1);
    T = tmprod(G,A,1:mode);

    r_err(r) = frob(X-T)/frob(X);
end 

%% STHOSVD and Error bound
for r = 1:n
    [G,A] = sthosvd(X,[r,r,r,r,r],order);
    T = tmprod(G,A,order);

    st_err(r) = frob(X-T)/frob(X);
    
    bound(r) = sqrt(1+r/(p-1))*st_err(r); %computes error bound
end

%% RSTHOSVD
for r = 1:n
    [G,A] = randsthosvd(X,[r,r,r,r,r],order,p,maxiter);
    T = tmprod(G,A,order);

    rst_err(r) = frob(X-T)/frob(X);
end


%%
figure,

subplot(1,2,1)
semilogy(1:n,h_err,'m','linewidth',2), hold on
semilogy(1:n,st_err,'b-.','linewidth',2)
semilogy(1:n,r_err,'r--','linewidth',2)
semilogy(1:n,rst_err,'k:','linewidth',2.5)
legend('HOSVD','STHOSVD','R-HOSVD','R-STHOSVD')
title('Hilbert Tensor Error')
xlabel('Target rank'), ylabel('Relative Error')
set(gca,'fontsize',16)
 
subplot(1,2,2)
semilogy(1:n,r_err,'r--','linewidth',2), hold on
semilogy(1:n,rst_err,'k:','linewidth',2.5)
semilogy(1:n,bound,'b','linewidth',2)
legend('R-HOSVD','R-STHOSVD','Error Bound')
title('Error Bound comparison')
xlabel('Target rank'), ylabel('Relative Error')
set(gca,'fontsize',16)


