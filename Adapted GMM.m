load('H:\AGMM FEATURES\MBH_AVENUE_TEST.mat')
load('H:\AGMM FEATURES\MBH_AVENUE_TRAIN.mat')
c1= cellfun(@transpose,MBH_Col_Train,'UniformOutput',false); %feature extraction
c1 = cellfun(@(x) normr(x), c1,'UniformOutput',false);
c2 = cellfun(@transpose,MBH_Col_Test,'UniformOutput',false);
c2 = cellfun(@(x) normr(x), c2,'UniformOutput',false);

%c = [c1 c2];
ncha=1;
 nWorkers=1;
nmix=16;
 map_tau = 16;
final_niter = 10;
ds_factor = 1;
for i=1:306
    fprintf('i= %d\n',i);
%     %traindata{i}=featCls{i}';
%gmm{i} = gmm_em(mat2cell(a{1,i}(:)), nmix, final_niter, ds_factor, nWorkers);
 gmm = gmm_em(c1(:,i), nmix, final_niter, ds_factor, nWorkers);
end

config = 'mvw';
gmmf=cell(1,306);
%for s=1:4
 %   fprintf('s= %d\n',s);
for z=1:306
     fprintf('z= %d\n',z);
gmmf{z} = mapAdapt(c1(:,z), gmm, map_tau, config); %
end
%  
     logLikT = zeros(306,306);
    %train=[];
    for i = 1:306 %no of train examples
    fprintf('i= %d\n',i);
    for j = 1:306
        %no of sub models
    gmm_llk1 = compute_llk(c1{i}, gmm.mu, gmm.sigma, gmm.w(:));
    gmm1_llk1 = compute_llk(c1{i}, gmmf{j}.mu, gmmf{j}.sigma, gmmf{j}.w(:));
    logLikT(i,j) = mean(gmm1_llk1 - gmm_llk1);
    %logLikT(i,j) = mean(gmm1_llk1);
    end
    end
 logLikTT = zeros(306,306);
for i = 1:306
fprintf('i=%d\n',i);
for j = 1:306
gmm2_llk1 = compute_llk(c2{i}, gmm.mu, gmm.sigma, gmm.w(:));
gmm3_llk1 = compute_llk(c2{i}, gmmf{j}.mu, gmmf{j}.sigma, gmmf{j}.w(:));
logLikTT(i,j) = mean(gmm3_llk1 - gmm2_llk1);
    end
%logLikTT(i,j) = mean(gmm3_llk1);
end
