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


LH_names{1} = 'LH-train';
LH_names{2} = 'LH-eval';
LH_names{3} = 'LH-fake1(sigma=1)';
LH_names{4} = 'LH-fake2(sigma=rand)';
LH_names{5} = 'LH-fake3(sigma=trainData)';

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


% nexttile
% LH_evaluation_vs_others = [];
% for i=1:size(LH_all_CBTs,2)
%     LH_evaluation_vs_others = [LH_evaluation_vs_others frobenious(LH_eval_view,LH_all_CBTs{i},LH_eval_nv)];
% end
% %frobenious distance between CBTs and evaluation samples
% bar(LH_evaluation_vs_others)
% ylim([0 40]);
% xlabel('Dataset Name');
% ylabel('Distance between Evaluation Samples-CBTs');
% title('Frobenious Distance Between CBTs and Evaluation Dataset');
% set(gca, 'XTickLabel',LH_names);


% nexttile
LH_cross_dists = [];
for i=1:size(LH_all_views,2)
    LH_cross_dists = [LH_cross_dists cross_distance(LH_eval_view,LH_eval_N,LH_all_views{i},LH_eval_N,LH_eval_nv)];
end
% %cross distance between new samples and evaluation samples
% bar(LH_cross_dists)
% ylim([0 40]);
% xlabel('Dataset Name');
% ylabel('Distance between Evaluation and Fake Samples');
% title('Cross Distance Between Fake and Evaluation Dataset');
% set(gca, 'XTickLabel',LH_names);

annotation('textbox', [0.945, 0.9, 0.1, 0.1], 'String', "# view = "+4,'FitBoxToText','on')
annotation('textbox', [0.945, 0.86, 0.1, 0.1], 'String', "# ROI = "+35,'FitBoxToText','on')
annotation('textbox', [0.945, 0.82, 0.1, 0.1], 'String', "# subject = "+36,'FitBoxToText','on')


%% Genetic Algorithm for finding optimum sigma value for generater
numIteration = 10;
[best_sigmas1, sigma_distances1, newViews1] = geneticAlgorithm(LH_eval_view,numIteration,LH_train_N,LH_train_nv,LH_train_numROIs,LH_train_representative_tensor,LH_train_NumFeatures,'half','version1','gen1');
[best_sigmas2, sigma_distances2, newViews2] = geneticAlgorithm(LH_eval_view,numIteration,LH_train_N,LH_train_nv,LH_train_numROIs,LH_train_representative_tensor,LH_train_NumFeatures,'mix','version1','gen1');

[best_sigmas3, sigma_distances3, newViews3] = geneticAlgorithm(LH_eval_view,numIteration,LH_train_N,LH_train_nv,LH_train_numROIs,LH_train_representative_tensor,LH_train_NumFeatures,'half','version2','gen1');
[best_sigmas4, sigma_distances4, newViews4] = geneticAlgorithm(LH_eval_view,numIteration,LH_train_N,LH_train_nv,LH_train_numROIs,LH_train_representative_tensor,LH_train_NumFeatures,'mix','version2','gen1');

for i=6:numIteration+5
    LH_names{i} = "itr-"+(i-5); 
end


%% bar plot start
LH_cross_dists2 = LH_cross_dists;
LH_cross_dists3 = LH_cross_dists;
LH_cross_dists4 = LH_cross_dists;

LH_cross_dists = [LH_cross_dists sigma_distances1];
LH_cross_dists2 = [LH_cross_dists2 sigma_distances2];
LH_cross_dists3 = [LH_cross_dists3 sigma_distances3];
LH_cross_dists4 = [LH_cross_dists4 sigma_distances4];

tot=[];
for i=1:length(LH_cross_dists)
    line = [LH_cross_dists(i) LH_cross_dists2(i) LH_cross_dists3(i) LH_cross_dists4(i)];
    tot = [tot; line];
end

b = bar(tot)
ylim([0 40]);
xlim([0 length(LH_names)+1]);
xlabel('Dataset Name');
ylabel('Distance between Evaluation and Fake Samples');
title('Cross Distance Between Fake and Evaluation Dataset');
% set legends
bar_names = {'psd-v1,half','psd-v1,mix','psd-v2,half','psd-v2,mix',};
set(b, {'DisplayName'}, bar_names') 
legend() 
set(gca, 'XTick', 1:length(LH_names),'XTickLabel',LH_names);
xtickangle(45)

% first bar naming
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1,ytips1,labels1,'VerticalAlignment','bottom',...
    'VerticalAlignment','bottom','rotation',90)
% second bar naming
xtips2 = b(2).XEndPoints;
ytips2 = b(2).YEndPoints;
labels2 = string(b(2).YData);
text(xtips2,ytips2,labels2,'VerticalAlignment','bottom',...
    'VerticalAlignment','bottom','rotation',90)
% third bar naming
xtips3 = b(3).XEndPoints;
ytips3 = b(3).YEndPoints;
labels3 = string(b(3).YData);
text(xtips3,ytips3,labels3,'VerticalAlignment','bottom',...
    'VerticalAlignment','bottom','rotation',90)
% fourth bar naming
xtips4 = b(4).XEndPoints;
ytips4 = b(4).YEndPoints;
labels4 = string(b(4).YData);
text(xtips4,ytips4,labels4,'VerticalAlignment','bottom',...
    'VerticalAlignment','bottom','rotation',90)
%% bar plot end


%% self frob distances
self_dists = [];
for i=1:5
    self_dists = [self_dists LH_all_self_frob_dists{i}];
end

%netNorm generates the fake representative tensors and CBTs
tot2=[];
for i=1:5
    line = [self_dists(i) self_dists(i) self_dists(i) self_dists(i) ];
    tot2 = [tot2; line];
end
for i=6:numIteration+5
    [LH_fake_NumFeatures6,self_dist1,LH_fake_representative_tensor6,LH_fake_netNorm_CBT6] = netNorm_func(newViews1{i-5},LH_train_nv,LH_train_N,LH_train_numROIs);
    [LH_fake_NumFeatures6,self_dist2,LH_fake_representative_tensor6,LH_fake_netNorm_CBT6] = netNorm_func(newViews2{i-5},LH_train_nv,LH_train_N,LH_train_numROIs);
    [LH_fake_NumFeatures6,self_dist3,LH_fake_representative_tensor6,LH_fake_netNorm_CBT6] = netNorm_func(newViews3{i-5},LH_train_nv,LH_train_N,LH_train_numROIs);
    [LH_fake_NumFeatures6,self_dist4,LH_fake_representative_tensor6,LH_fake_netNorm_CBT6] = netNorm_func(newViews4{i-5},LH_train_nv,LH_train_N,LH_train_numROIs);

    line = [self_dist1 self_dist2 self_dist3 self_dist4];
    tot2 = [tot2; line];
end

b2 = bar(tot2)
ylim([0 40]);
xlim([0 length(LH_names)+1]);
xlabel('Dataset Name');
ylabel('Self Frobenious Distance');
title('Frobenious Distance Between CBT and samples');
% set legends
bar_names = {'psd-v1,half','psd-v1,mix','psd-v2,half','psd-v2,mix',};
set(b2, {'DisplayName'}, bar_names') 
legend() 
set(gca, 'XTick', 1:length(LH_names),'XTickLabel',LH_names);
xtickangle(45)

% first bar naming
xtips1 = b2(1).XEndPoints;
ytips1 = b2(1).YEndPoints;
labels1 = string(b2(1).YData);
text(xtips1,ytips1,labels1,'VerticalAlignment','bottom',...
    'VerticalAlignment','bottom','rotation',90)
% second bar naming
xtips2 = b2(2).XEndPoints;
ytips2 = b2(2).YEndPoints;
labels2 = string(b2(2).YData);
text(xtips2,ytips2,labels2,'VerticalAlignment','bottom',...
    'VerticalAlignment','bottom','rotation',90)
% third bar naming
xtips3 = b2(3).XEndPoints;
ytips3 = b2(3).YEndPoints;
labels3 = string(b2(3).YData);
text(xtips3,ytips3,labels3,'VerticalAlignment','bottom',...
    'VerticalAlignment','bottom','rotation',90)
% fourth bar naming
xtips4 = b2(4).XEndPoints;
ytips4 = b2(4).YEndPoints;
labels4 = string(b2(4).YData);
text(xtips4,ytips4,labels4,'VerticalAlignment','bottom',...
    'VerticalAlignment','bottom','rotation',90)
%% self frob distances end

%% sigma value for each feature method-2

%% sigma value for each feature method-2 end

