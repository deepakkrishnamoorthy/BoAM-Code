

train1 = cellfun(@transpose,train1,'UniformOutput',false);
train1 = cellfun(@(x) normr(x), train1,'UniformOutput',false);%traindata
train2 = cellfun(@transpose,train2,'UniformOutput',false);
train2 = cellfun(@(x) normr(x), train2,'UniformOutput',false);%traindata
test = cellfun(@transpose,test,'UniformOutput',false);
test = cellfun(@(x) normr(x), test,'UniformOutput',false);%traindata
train = cellfun(@transpose,train,'UniformOutput',false);
train = cellfun(@(x) normr(x), train,'UniformOutput',false);%traindata

%numCls=15;
ncha=2;
   nWorkers=1;
 %nmix = 4;           % In this case, we know the # of mixtures needed
 %nmix=8;
 nmix=32;
final_niter = 10;
ds_factor = 1;
ubm = gmm_em(train(:), nmix, final_niter, ds_factor, nWorkers);%creation of UBM For all features

map_tau = 16; %relavance factor 10
 config = 'mvw';
%gmm=cell(1,numCls);
%for s=1:numCls
gmm = mapAdapt(train1, ubm, map_tau, config); %adapting invidual classes with UBM
%end
gmm1=mapAdapt(train2, ubm, map_tau, config); %adapting invidual classes with UBM
%end
gmmc={gmm,gmm1};
 for i = 1:150
    for j = 1:2
        logllk = compute_llk1(test{i},gmmc{j}.mu,gmmc{j}.sigma,gmmc{j}.w(:));
        logLik(i,j) = mean(logllk);
        %[llkVal, llkLabel]=max(logLik,[],2);
    end
 end
 
 for i=1:150
[logval(i) llkLabel(i)] = max(logLik(i,:),[],2);  
 end
actuallabel=[ones(66,1);2*ones(84,1)];
acc = sum(llkLabel' == actuallabel) ./ numel(actuallabel) * 100 ;

    

