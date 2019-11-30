clear all
clc
folder='G:\MATLAB Runtime\v85\dcase2016\fold1dcase16\';
%folder = 'E:\User-Files\Documents\MATLAB Runtime\v85\ieeeaasp dataset\fold1\train test 1\';
 files = dir(strcat(folder,'*.wav'));
for i = 1:length(files)
     %for i=1:2
     fname = strcat(folder,files(i,1).name);
     [d sr] = audioread(fname);
     %features{i} = melcepst(d,sr,'C',13);
     features{i} = melcepst1(d,sr,'C',20);
    % features{i} = melcepst(d,sr);
     fprintf('i= %d\n',i);
     a = cellfun(@transpose,features,'UniformOutput',false); %feature extraction
end
 k = 1;
for i = 1: 15
    temp=[];
    for j = 1:45
        temp=[temp;features{k}];
        k=k+1;
    end
    featCls{i}=temp;
     a = cellfun(@transpose,featCls,'UniformOutput',false);%traindata
    %data=traindata';
end
 numCls=15;
ncha=2;
   nWorkers=1;
 nmix = 32;          % In this case, we know the # of mixtures needed
final_niter = 10;
ds_factor = 1;
%ubm=cell(1,numCls);
for i=1:numCls
    %traindata{i}=featCls{i}';
    fprintf('i= %d\n',i);
gmm{i} = gmm_em(a(:,i), nmix, final_niter, ds_factor, nWorkers);
end
testData=[];
for i = 1:290
    testData{i} = features{i+675};
   t = cellfun(@transpose,testData,'UniformOutput',false);%testdata
    %t=testData';
end
%gmmScores = score_gmm_trials(gmm, reshape(t',[ numCls*ncha,1]), trials, ubm);
 for i = 1:290
    for j = 1:15
        logllk = compute_llk1(t{i},gmm{j}.mu,gmm{j}.sigma,gmm{j}.w(:));
        logLik(i,j) = mean(logllk);
        %[llkVal, llkLabel]=max(logLik,[],2);
    end
 end
for i=1:290
[logval(i) llkLabel(i)] = max(logLik(i,:),[],2);  
end
actuallabel=[ones(19,1);2*ones(19,1);3*ones(18,1);4*ones(21,1);5*ones(18,1);6*ones(21,1);7*ones(19,1);8*ones(22,1);9*ones(21,1);10*ones(19,1);11*ones(18,1);12*ones(21,1);13*ones(19,1);14*ones(18,1);15*ones(17,1)];
 acc = sum(llkLabel' == actuallabel) ./ numel(actuallabel) * 100 ;
 C=confusionmat(actuallabel,llkLabel)
