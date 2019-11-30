
%numCls=15;
ncha=2;
   nWorkers=1;
 nmix = 32;          % In this case, we know the # of mixtures needed
final_niter = 10;
ds_factor = 1;
%ubm=cell(1,numCls);

    %traindata{i}=featCls{i}';
  
gmm0 = gmm_em(train1(:), nmix, final_niter, ds_factor, nWorkers);
gmm1=gmm_em(train2(:), nmix, final_niter, ds_factor, nWorkers);
gmm={gmm0,gmm1};


%gmmScores = score_gmm_trials(gmm, reshape(t',[ numCls*ncha,1]), trials, ubm);
 for i = 1:158
    for j = 1:2
        logllk = compute_llk1(test{i},gmm{j}.mu,gmm{j}.sigma,gmm{j}.w(:));
        logLik(i,j) = mean(logllk);
        %[llkVal, llkLabel]=max(logLik,[],2);
    end
 end
for i=1:158
[logval(i) llkLabel(i)] = max(logLik(i,:),[],2);  
end
actuallabel=[ones(66,1);2*ones(92,1)];
acc = sum(llkLabel' == actuallabel) ./ numel(actuallabel) * 100 ;
