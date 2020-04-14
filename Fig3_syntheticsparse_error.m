%% Fig3_syntheticsparse_error.m
%
% This code generates Figure 3 from the paper
%   'Randomized Algorithms for Low-rank Tensor
%    Decompositions in the Tucker Format'
%       - Minster, Saibaba, Kilmer

%fixes random seed
seed = rng;
rng(seed);

%tensor parameters
n = 200;
sparsity = .05;   %percentage of nonzeros
gap = [2,10,200];

%parameters for algorithms
order = [1,2,3]; % processing order
p = 5; % oversampling parameter
maxiter = 0; %number of subspace iterations
rank = 25; %target rank

figure
for v = 1:3
    % form tensor
    X = cpsparse_tensor(n,sparsity,gap(v),n);

    Q = sptensor(X);
    mode = length(size(X)); 


    k = 0;
    relerr = zeros((n-p)/5,1);
    sterr = zeros((n-p)/5,1);
    rsterr = zeros((n-p)/5,1);

    for r = 1:5:n-p
        k = k + 1;
       
        % SP-STHOSVD
        [G,A,~] = sparse_tucker(Q,[r,r,r],5,order);
        T = ttm(G,A,1:mode);
        T = tensor(T);
        relerr(k) = norm(Q-T)/norm(Q);

        % STHOSVD
        [G2,A2] = sthosvd(X,[r+5,r+5,r+5],order);
        T2 = tmprod(G2,A2,1:mode);
        sterr(k) = frob(X-T2)/frob(X);

        % R-STHOSVD
        [G3,A3] = randsthosvd(X,[r+5,r+5,r+5],order,p,maxiter);
        T3 = tmprod(G3,A3,1:mode);
        rsterr(k) = frob(X-T3)/frob(X);

     end

subplot(1,3,v)
semilogy(1:5:n-p,relerr,'k-', 'Linewidth',1.5), hold on
semilogy(1:5:n-p,sterr,'b:','Linewidth',2)
semilogy(1:5:n-p,rsterr,'g--','Linewidth',1.5)
legend('SP-STHOSVD','STHOSVD','R-STHOSVD','Location','southwest')
title(['$\gamma$ = ', num2str(gap(v))],'interpreter','latex')
set(gca,'fontsize',16)
xlabel('Rank')
ylabel('Relative error')

end
