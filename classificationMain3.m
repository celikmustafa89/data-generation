clc
clear
%load data
files = dir('./data/Data_77subjects/LH77subjects');
for i=1:77
    a=load(strcat("./data/Data_77subjects/LH77subjects/subject"+i+".mat"));
    trainTensor{i} = a.A;
end
trainy = load("./data/Data_77subjects/labels77.mat");
trainy = trainy.labels;

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

Time = [];
tic
modelNames = {};
[accuracy1,specificity1,sensitivity1] = real_data_pca(X_AD_tensors,X_MCI_tensors,1)
result{1} = sum(accuracy1)/length(accuracy1);
modelNames{1} = "only real sample (N)";
Time(1) = toc;
disp('real_data_pca model is done.')


% sigma = 1
tic
[accuracy2,specificity2,sensitivity2] = sigma1_pca(X_AD_tensors,X_MCI_tensors,'fake1',1);
result{2} = sum(accuracy2)/length(accuracy2);
modelNames{2} = "sigma = 1 (N+5N)";
Time(2) = toc;
disp('sigma=1 model is done.')

% sigma = random
tic
[accuracy3,specificity3,sensitivity3] = sigma1_pca(X_AD_tensors,X_MCI_tensors,'fake2',1);
result{3} = sum(accuracy3)/length(accuracy3);
modelNames{3} = "sigma = random (N+5N)";
Time(3) = toc;
disp('sigma=random model is done.')

% sigma = cov
tic
[accuracy4,specificity4,sensitivity4] = sigma1_pca(X_AD_tensors,X_MCI_tensors,'cov',1);
result{4} = sum(accuracy4)/length(accuracy4);
modelNames{4} = "sigma = real sample covariance (N+5N)";
Time(4) = toc;
disp('sigma=covariance model is done.')

% sigma genetic
tic
[accuracy5,specificity5,sensitivity5] = genetic_pca(X_AD_tensors,X_MCI_tensors,0);
toc
result{5} = sum(accuracy5)/length(accuracy5);
modelNames{5} = "sigma = genetic (N+5N)";
Time(5) = toc;
disp('sigma=genetic model is done.')

acc = cell2mat(result);
figure(1)
color={};
color{1}="#D95319"; % red
color{2}="#EDB120";	% yellow
color{3}="#77AC30"; % green
color{4}="#7E2F8E";	% mor
color{5}="#0072BD";	% blue
hold on;
for i=1:5
    b=bar(i,acc(i));
    b.FaceColor = color{i};
    xtips1 = b(1).XEndPoints;
    ytips1 = b(1).YEndPoints;
    labels1 = string(b(1).YData);
    text(xtips1,ytips1,labels1,'VerticalAlignment','bottom',...
    'VerticalAlignment','bottom','rotation',45)
end
xlabel('Models')
ylabel('Accuracy')
title('Model Comparisons Left-Hemisphere')
ylim([0 1]);
plot([0 6],[acc(5) acc(5)], ':','color',[0 0.4470 0.7410],'linewidth',2)
legend(modelNames)
grid on
hold off;

% b=bar(acc)
% xlabel('models')
% ylabel('accuracy')
% title('model comparisons')
% ylim([0 1]);
% xlim([0 length(acc)+1]);
% set(gca, 'XTick', 1:length(modelNames),'XTickLabel',modelNames);
% xtickangle(45)
% % first bar naming
% xtips1 = b(1).XEndPoints;
% ytips1 = b(1).YEndPoints;
% labels1 = string(b(1).YData);
% text(xtips1,ytips1,labels1,'VerticalAlignment','bottom',...
%     'VerticalAlignment','bottom','rotation',45)