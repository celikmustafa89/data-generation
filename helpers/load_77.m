function [X_AD_tensor_folds,X_MCI_tensor_folds,N,fold_size,fold_k,nv,ROIs] = load_77()
%load data
files = dir('./data/Data_77subjects/LH77subjects');
for i=1:77
    a=load(strcat("./data/Data_77subjects/LH77subjects/subject"+i+".mat"));
    trainTensor{i} = a.A;
end
trainy = load("./data/Data_77subjects/labels77.mat");
trainy = trainy.labels;

% get details of data
[~,ROIs,nv] = size(trainTensor{1});

%% create dataset
% 35 AD samples
% 35 MCI samples
X_AD_tensors = {};
for i=1:35 % get 35 AD data samples
    X_AD_tensors{i}=trainTensor{i};
end
X_MCI_tensors = {};
for i=43:77 % get 35 MCI data samples
    X_MCI_tensors{i-42}=trainTensor{i};
end



%% create folds for cross-validation
fold_k=5;
[~,N]=size(X_AD_tensors);
fold_size=N/fold_k;
for i=1:N
    foldnum = fix((i-1)/fold_size) + 1;
    indis = rem(i-1,fold_size)+1;
   
    X_AD_tensor_folds{foldnum}{indis} = X_AD_tensors{i};
    X_MCI_tensor_folds{foldnum}{indis} = X_MCI_tensors{i};
end
