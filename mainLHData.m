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

% generate CBTs from train data
[LH_train_NumFeatures,LH_train_Frob_dist,LH_train_representative_tensor,LH_train_netNorm_CBT] = netNorm_func(LH_train_view,LH_train_nv,LH_train_N,LH_train_numROIs);
LH_all_views{1} = LH_train_view;
LH_all_CBTs{1} = LH_train_netNorm_CBT;
LH_all_self_frob_dists{1} = LH_train_Frob_dist;

%generates CBTs from eval data
[LH_eval_NumFeatures,LH_eval_Frob_dist,LH_eval_representative_tensor,LH_eval_netNorm_CBT] = netNorm_func(LH_eval_view,LH_eval_nv,LH_eval_N,LH_eval_numROIs);
LH_all_views{2} = LH_eval_view;
LH_all_CBTs{2} = LH_eval_netNorm_CBT;
LH_all_self_frob_dists{2} = LH_eval_Frob_dist;


%generate fake samples (fake1=sigma1)
[LH_fake_samples1,LH_fake_view1] = generateSample(LH_train_view,LH_train_representative_tensor,LH_train_N,LH_train_nv,LH_train_NumFeatures,LH_train_numROIs,'fake1');
%netNorm generates the fake representative tensors and CBTs
[LH_fake_NumFeatures1,LH_fake_Frob_dist1,LH_fake_representative_tensor1,LH_fake_netNorm_CBT1] = netNorm_func(LH_fake_view1,LH_train_nv,LH_train_N,LH_train_numROIs);
LH_all_views{3} = LH_fake_view1;
LH_all_CBTs{3} = LH_fake_netNorm_CBT1;
LH_all_self_frob_dists{3} = LH_fake_Frob_dist1;

%generate fake samples (fake2=sigmarand)
[LH_fake_samples2,LH_fake_view2] = generateSample(LH_train_view,LH_train_representative_tensor,LH_train_N,LH_train_nv,LH_train_NumFeatures,LH_train_numROIs,'fake2');
%netNorm generates the fake representative tensors and CBTs
[LH_fake_NumFeatures2,LH_fake_Frob_dist2,LH_fake_representative_tensor2,LH_fake_netNorm_CBT2] = netNorm_func(LH_fake_view2,LH_train_nv,LH_train_N,LH_train_numROIs);
LH_all_views{4} = LH_fake_view2;
LH_all_CBTs{4} = LH_fake_netNorm_CBT2;
LH_all_self_frob_dists{4} = LH_fake_Frob_dist2;

%generate fake samples (fake3=sigma=trainData)
[LH_fake_samples3,LH_fake_view3] = generateSample(LH_train_view,LH_train_representative_tensor,LH_train_N,LH_train_nv,LH_train_NumFeatures,LH_train_numROIs,'cov');
%netNorm generates the fake representative tensors and CBTs
[LH_fake_NumFeatures3,LH_fake_Frob_dist3,LH_fake_representative_tensor3,LH_fake_netNorm_CBT3] = netNorm_func(LH_fake_view3,LH_train_nv,LH_train_N,LH_train_numROIs);
LH_all_views{5} = LH_fake_view3;
LH_all_CBTs{5} = LH_fake_netNorm_CBT3;
LH_all_self_frob_dists{5} = LH_fake_Frob_dist3;


LH_names{1} = 'LHtrain';
LH_names{2} = 'LHeval';
LH_names{3} = 'LHfake1(sigma=1)';
LH_names{4} = 'LHfake2(sigma=rand)';
LH_names{5} = 'LHfake3(sigma=trainData)';

tiledlayout(2,2) % Requires R2019b or later

nexttile
%frobenious distance between CBTs and it's sample space
bar(cell2mat(LH_all_self_frob_dists))
ylim([0 40]);
xlabel('Dataset Name');
ylabel('Self CBT Distance');
title('Frobenious Distance Between CBTs and Its Dataset');
set(gca, 'XTickLabel',LH_names);


nexttile
LH_original_vs_others = [];
for i=1:size(LH_all_CBTs,2)
    LH_original_vs_others = [LH_original_vs_others frobenious(LH_train_view,LH_all_CBTs{i},LH_eval_nv)];
end
%frobenious distance between CBTs and original samples
bar(LH_original_vs_others)
ylim([0 40]);
xlabel('Dataset Name');
ylabel('Distance between Train Samples-CBTs');
title('Frobenious Distance Between CBTs and Train Dataset');
set(gca, 'XTickLabel',LH_names);


nexttile
LH_evaluation_vs_others = [];
for i=1:size(LH_all_CBTs,2)
    LH_evaluation_vs_others = [LH_evaluation_vs_others frobenious(LH_eval_view,LH_all_CBTs{i},LH_eval_nv)];
end
%frobenious distance between CBTs and evaluation samples
bar(LH_evaluation_vs_others)
ylim([0 40]);
xlabel('Dataset Name');
ylabel('Distance between Evaluation Samples-CBTs');
title('Frobenious Distance Between CBTs and Evaluation Dataset');
set(gca, 'XTickLabel',LH_names);


nexttile
LH_cross_dists = [];
for i=1:size(LH_all_views,2)
    LH_cross_dists = [LH_cross_dists cross_distance(LH_eval_view,LH_eval_N,LH_all_views{i},LH_eval_N,LH_eval_nv)];
end
%cross distance between new samples and evaluation samples
bar(LH_cross_dists)
ylim([0 40]);
xlabel('Dataset Name');
ylabel('Distance between Evaluation and Fake Samples');
title('Cross Distance Between Fake and Evaluation Dataset');
set(gca, 'XTickLabel',LH_names);

annotation('textbox', [0.945, 0.9, 0.1, 0.1], 'String', "# view = "+4,'FitBoxToText','on')
annotation('textbox', [0.945, 0.86, 0.1, 0.1], 'String', "# ROI = "+35,'FitBoxToText','on')
annotation('textbox', [0.945, 0.82, 0.1, 0.1], 'String', "# subject = "+36,'FitBoxToText','on')