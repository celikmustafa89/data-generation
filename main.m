
clc
clear all;
close all;

%data definition as simulated data
fprintf("Enter inputs for creating simulated train data:\n");
[original_view,original_nv,original_N,original_numROIs]=simulateData;
fprintf("Simulated train data is created.\n\n");
all_views{1} = original_view;

% generates the representative tensors and CBTs (original)
[original_NumFeatures,original_Frob_dist,original_representative_tensor,original_netNorm_CBT] = netNorm_func(original_view,original_nv,original_N,original_numROIs);
all_CBTs{1} = original_netNorm_CBT;
all_self_frob_dists{1} = original_Frob_dist;

%generate fake samples (fake1)
[fake_samples1,fake_view1] = generateSample(original_representative_tensor,original_Noriginal_,original_nv,original_NumFeatures,original_numROIs);
%netNorm generates the fake representative tensors and CBTs
[fake_NumFeatures1,fake_Frob_dist1,fake_representative_tensor1,fake_netNorm_CBT1] = netNorm_func(fake_view1,original_nv,original_N,original_numROIs);
all_views{2} = fake_view1;
all_CBTs{2} = fake_netNorm_CBT1;
all_self_frob_dists{2} = fake_Frob_dist1;

%generate fake samples (fake2)
[fake_samples2,fake_view2] = generateSample(original_representative_tensor,original_N,original_nv,original_NumFeatures,original_numROIs);
%netNorm generates the fake representative tensors and CBTs
[fake_NumFeatures2,fake_Frob_dist2,fake_representative_tensor2,fake_netNorm_CBT2] = netNorm_func(fake_view2,original_nv,original_N,original_numROIs);
all_views{3} = fake_view2;
all_CBTs{3} = fake_netNorm_CBT2;
all_self_frob_dists{3} = fake_Frob_dist2;

%data definition for evaluation
fprintf("Enter inputs for creating simulated evaluation data:\n");
[eval_view,eval_nv,eval_N,eval_numROIs]=simulateData;
fprintf("simulated evaluation data is created.\n\n");

%netNorm generates the evaluation representative tensors and CBTs
[eval_NumFeatures,eval_Frob_dist,eval_representative_tensor,eval_netNorm_CBT] = netNorm_func(eval_view,eval_nv,eval_N,eval_numROIs);
all_views{4} = eval_view;
all_CBTs{4} = eval_netNorm_CBT;
all_self_frob_dists{4} = eval_Frob_dist;

names{1} = 'original';
names{2} = 'fake1';
names{3} = 'fake2';
names{4} = 'evaluation';

tiledlayout(2,2) % Requires R2019b or later

nexttile
%frobenious distance between CBTs and it's sample space
bar(cell2mat(all_self_frob_dists))
ylim([0 40]);
xlabel('Dataset Name');
ylabel('Self CBT Distance');
title('Frobenious Distance Between CBTs and Its Dataset');
set(gca, 'XTickLabel',names);

nexttile
original_vs_others = [];
for i=1:size(all_CBTs,2)
    original_vs_others = [original_vs_others frobenious(original_view,all_CBTs{i},nv)];
end
%frobenious distance between CBTs and original samples
bar(original_vs_others)
ylim([0 40]);
xlabel('Dataset Name');
ylabel('Distance between Original Samples-CBTs');
title('Frobenious Distance Between CBTs and Original Dataset');
set(gca, 'XTickLabel',names);

nexttile
evaluation_vs_others = [];
for i=1:size(all_CBTs,2)
    evaluation_vs_others = [evaluation_vs_others frobenious(eval_view,all_CBTs{i},nv)];
end
%frobenious distance between CBTs and original samples
bar(evaluation_vs_others)
ylim([0 40]);
xlabel('Dataset Name');
ylabel('Distance between Evaluation Samples-CBTs');
title('Frobenious Distance Between CBTs and Evaluation Dataset');
set(gca, 'XTickLabel',names);

nexttile
cross_dists = [];
for i=1:size(all_views,2)
    cross_dists = [cross_dists cross_distance(eval_view,N,all_views{i},N,nv)];
end
%cross distance between new samples and evaluation samples
bar(cross_dists)
ylim([0 40]);
xlabel('Dataset Name');
ylabel('Distance between Evaluation and Fake Samples');
title('Cross Distance Between Fake and Evaluation Dataset');
set(gca, 'XTickLabel',names);
