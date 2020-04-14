%% Tab6_enron.m
%
% This code generates Table 6 from the paper
%   'Randomized Algorithms for Low-rank Tensor
%    Decompositions in the Tucker Format'
%       - Minster, Saibaba, Kilmer
%
% Requires enron.tns from FROSTT database: http://frostt.io/

%% load and subsample
load enron.tns 
M = enron;

% convert to tensor
subs = M(:,1:4);
vals = M(:,5);
Q = sptensor(subs,vals);

% condense to smaller size and subsample
T = collapse(Q,4);
B = T(1:15:end,1:15:end,1:20:end);
Bnrm = norm(B);
B2 = double(B); %format for R-STHOSVD

seed = rng;
rng(seed);

clear enron
clear M

%% variables
order = [3,1,2]; 
d = 3;

% initialize
int = 20:25:195;
l = length(int);
t_sp = zeros(1,l);
t_rst = zeros(1,l);
err_sp = zeros(1,l);
err_rst = zeros(1,l);

%% 
for i = 1:l
    r = int(i);
    
    % SP-STHOSVD Runtime
    for j = 1:3
        tic; [G,A] = sparse_tucker(Q,r,5,order); t = toc;
        t_sp(i) = t_sp(i)+t;
    end
    
    % SP-STHOSVD Error
    S = ttm(G,A,order);
    S = tensor(S);
    err_sp(i) = norm(B-S)/Bnrm;
    
    %R-STHOSVD Runtime
    for j = 1:3
        tic; [G,A] = randsthosvd(B2,r,order,5,0); t = toc;
        t_rst(i) = t_rst(i)+t;
    end
    
    %R-STHOSVD Error
    S2 = tmprod(G,A,order);
    err_rst(i) = frob(B2-S2)/frob(B2);
end
t_sp = t_sp./3;
t_rst = t_rst./3;

%% display
disp('Runtime for SP-STHOSVD:')
disp(t_sp)

disp('Error for SP-STHOSVD:')
disp(err_sp);

disp('Runtime for R-STHOSVD:')
disp(t_rst)

disp('Error for R-STHOSVD:')
disp(err_rst);
