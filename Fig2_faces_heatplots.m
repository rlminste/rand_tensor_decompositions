%% Fig2_faces_heatplots.m
%
% This code generates Figure 2 from the paper
%   'Randomized Algorithms for Low-rank Tensor
%    Decompositions in the Tucker Format'
%       - Minster, Saibaba, Kilmer
%
% Data from Olivetti Database of Faces, AT&T, 
%   https://cs.nyu.edu/~roweis/data.html


% load data
load olivettifaces.mat

% form tensor
X = zeros(40,4096,10);   %people x pixels x poses
for i = 1:40
    X(i,:,:) = faces(:,1+10*(i-1):10*i);
end


Xnrm = frob(X);
order = [2,1]; % processing order
k = 0; % target rank
p = 5; % oversampling parameter
q = 0; % number of subspace iterations


%initialize
int1 = 1:40;
int2 = 5:5:400;
l1 = length(int1);
l2 = length(int2);

err_st = zeros(l1,l2);
err_rst = zeros(l1,l2);


%% STHOSVD
for i = 1:l1
    for j = 1:l2
        [G,A] = sthosvd(X,[int1(i),int2(j),10],order);
        T = tmprod(G,A,order);
        err_st(i,j) = frob(X-T)/Xnrm;
    end
end

%% RSTHOSVD

for i = 1:l1
    for j = 1:l2
        [G,A] = randsthosvd(X,[int1(i),int2(j),10],order,p,q);
        T = tmprod(G,A,order);
        err_rst(i,j) = frob(X-T)/Xnrm;
    end
end

%% Adaptive R-STHOSVD

tol = [0.3,0.25,0.2,0.15,0.1,0.075];  
E = zeros(length(tol),2);

%R-STHOSVD
for i = 1:length(tol)
    [G,A,iter] = sthosvd_adaptrng(X,order,tol(i),5);
    E(i,:) = [size(G,2),size(G,1)];
end

%% Heat plots
figure,

subplot(1,2,1)
[X,Y] = meshgrid(int2,int1);
pcolor(X,Y,err_st(int1,int2)), hold on
shading interp  
caxis([0.02,0.4])
title('STHOSVD Relative Error')
xlabel('rank of pixels')
ylabel('rank of people')
set(gca,'fontsize',18)

subplot(1,2,2)
[X,Y] = meshgrid(int2,int1);
pcolor(X,Y,err_rst(int1,int2)), hold on
shading interp  
colorbar, caxis([0.02,0.4])
for j = 2:length(tol)
    plot(E(j,1),E(j,2),'w*','Markersize',10)
    str = ['\epsilon = ' num2str(tol(j))];
    text(E(j,1)+8,E(j,2)-1,str,'fontsize',16,'color','w')
end
title('RSTHOSVD Relative Error')
xlabel('rank of pixels')
ylabel('rank of people')
set(gca,'fontsize',18)
