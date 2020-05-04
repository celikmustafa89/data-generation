%% LH77 Subjects

clc
clear all;
close all;

%load data
files = dir('./data/Data_77subjects/LH77subjects');

%for i=3:floor(length(files)/2)+1
for i=3:20
    if ~strcmp(files(i).name,'.') & ~strcmp(files(i).name,'..')
        a=load(strcat("./data/Data_77subjects/LH77subjects/"+files(i).name));
        LH_train{i-2} = a.A;
    end
end
%for i=floor(length(files)/2)+2:length(files)-1
for i=21:38
    if ~strcmp(files(i).name,'.') & ~strcmp(files(i).name,'..')
        a=load(strcat("./data/Data_77subjects/LH77subjects/"+files(i).name));
        LH_eval{i-20} = a.A;
    end
end
%for i=3:floor(length(files)/2)+1
for i=42:59
    if ~strcmp(files(i).name,'.') & ~strcmp(files(i).name,'..')
        a=load(strcat("./data/Data_77subjects/LH77subjects/"+files(i).name));
        LH_train{i-23} = a.A;
    end
end
%for i=floor(length(files)/2)+2:length(files)-1
for i=60:77
    if ~strcmp(files(i).name,'.') & ~strcmp(files(i).name,'..')
        a=load(strcat("./data/Data_77subjects/LH77subjects/"+files(i).name));
        LH_eval{i-41} = a.A;
    end
end

[~,LH_train_N] = size(LH_train);
[~,LH_train_numROIs,LH_train_nv] = size(LH_train{1});

[~,LH_eval_N] = size(LH_eval);
[~,LH_eval_numROIs,LH_eval_nv] = size(LH_eval{1});

% generate LH_eval view for netnorm
for i=1:LH_eval_N
    for v=1:LH_eval_nv
        LH_eval_view{v}(:,:,i) = LH_eval{i}(:,:,v);
    end
end
% generate LH_train view for netnorm
for i=1:LH_train_N
    for v=1:LH_train_nv
        LH_train_view{v}(:,:,i) = LH_train{i}(:,:,v);
    end
end

[LH_train_NumFeatures,LH_train_Frob_dist,LH_train_representative_tensor,LH_train_netNorm_CBT] = netNorm_func(LH_train_view,LH_train_nv,LH_train_N,LH_train_numROIs);

NumFeatures = ((LH_eval_numROIs*LH_eval_numROIs-LH_eval_numROIs)/2);
nv = LH_train_nv;

%%%%

% vectorize representative tensors
tensors = LH_train_representative_tensor;
view_vec = {nv};
for k=1:nv
    view_vec{k} = vectorize(tensors{k});
end

