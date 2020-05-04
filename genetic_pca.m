function [accuracy,specificity,sensitivity] = genetic_pca(X_AD_tensors,X_MCI_tensors,drawGraph)

% get details of data
[~,ROIs,nv] = size(X_MCI_tensors{1});

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

accuracy = [];
specificity = [];
sensitivity = [];
datasetNames = {};

%tiledlayout(fix((fold_k+1)/2),2)
tiledlayout(3,2)
sigmasAD  = {};
sigmasMCI = {};
reptensorsAD  = {};
reptensorsMCI = {};
%load('/Users/mustafacelik/thesis/codes/mycodes/data-generation/LH_sigma_tensors.mat');   

for i=1:fold_k
    datasetNames{i} = append('LH_fold-',num2str(i));
    
    % divides test set and train set with fold manner
    [datasetTrain_AD,datasetTest_AD,datasetTrain_MCI,datasetTest_MCI] = get_fold(X_AD_tensor_folds,X_MCI_tensor_folds,fold_k,i);
   
    % vectorize features for train samples
    train_AD_r = cell2vec(datasetTrain_AD);
    train_MCI_r = cell2vec(datasetTrain_MCI);
    % vectorize features for test samples
    test_AD_r = cell2vec(datasetTest_AD);
    test_MCI_r = cell2vec(datasetTest_MCI);
    % combine two different populations(AD and MCI) in same test array
    testX = [test_AD_r ; test_MCI_r];
    testy = [ones(size(test_AD_r,1),1) ; zeros(size(test_MCI_r,1),1)];
    
    %% sample generation
    netnorm_tensor_ad = cell2tensor(datasetTrain_AD);
    netnorm_tensor_mci = cell2tensor(datasetTrain_MCI);
    trainN_AD = size(datasetTrain_AD,2);
    trainN_MCI = size(datasetTrain_MCI,2);
    [NumFeatures_AD ,Frob_dist_AD ,representative_tensor_AD ,netNorm_CBT_AD ] = netNorm_func(netnorm_tensor_ad,nv,trainN_AD,ROIs);
    [NumFeatures_MCI,Frob_dist_MCI,representative_tensor_MCI,netNorm_CBT_MCI] = netNorm_func(netnorm_tensor_mci,nv,trainN_MCI,ROIs);
    newN_AD=size(train_AD_r,1)*5;
    newN_MCI=size(train_MCI_r,1)*5;
%   % genetic algorithm and newViews of generated samples
%   % we are finding best sigma values for AD and MCI populations seperately.
%   % we will use these sigma values for creating fake datasets.
    [best_sigmasAD, sigma_distancesAD, newViews_AD]    = geneticAlgorithm(netnorm_tensor_ad,trainN_AD,nv,ROIs,representative_tensor_AD, NumFeatures_AD, 'half','version2','gen1');
    [best_sigmasMCI, sigma_distancesMCI, newViews_MCI] = geneticAlgorithm(netnorm_tensor_mci,trainN_MCI,nv,ROIs,representative_tensor_MCI,NumFeatures_MCI,'half','version2','gen1');
    sigmasAD{i}  = best_sigmasAD{1};
    sigmasMCI{i} = best_sigmasMCI{1};
    rep_tensorsAD{i}  = representative_tensor_AD;
    rep_tensorsMCI{i} = representative_tensor_MCI;
    saveSigmaAndTensor(append('fold-',num2str(i),'_',datestr(now),'.mat'),sigmasAD,sigmasMCI,rep_tensorsAD,rep_tensorsMCI);
    [~, fakeView_AD]  = generateSampleGA(rep_tensorsAD{i} ,newN_AD, nv,NumFeatures_AD, ROIs, sigmasAD{i});
    [~, fakeView_MCI] = generateSampleGA(rep_tensorsMCI{i},newN_MCI,nv,NumFeatures_MCI,ROIs, sigmasMCI{i});

    %load('/Users/mustafacelik/thesis/codes/mycodes/data-generation/saved/gold/LH_sigma_tensors.mat');    
    %[~, fakeView_AD]  = generateSampleGA(representative_tensor_AD{i} ,newN_AD, nv,NumFeatures_AD, ROIs, sigmas_AD{i});
    %[~, fakeView_MCI] = generateSampleGA(representative_tensor_MCI{i},newN_MCI,nv,NumFeatures_MCI,ROIs, sigmas_MCI{i});

    train_AD_fake = tensor2vec(fakeView_AD);
    train_MCI_fake = tensor2vec(fakeView_MCI);
    
    %% combine train sets
    % combine two different populations(AD and MCI) in same train array
    X = [train_AD_r ; train_AD_fake; train_MCI_r; train_MCI_fake];
    y = [ones(size(train_AD_r,1)+size(train_AD_fake,1),1) ; zeros(size(train_MCI_r,1)+size(train_MCI_fake,1),1)];
    
    %% model train  test
    [coeffs, scores, latents] = pca(X);
    cof = coeffs(:,1:2);
    X_pca = X*cof;
    testX_pca = testX*cof;
    
    % train model
    SVMModel = fitcsvm(X_pca,y);

    % test model
    [label,score] = predict(SVMModel,testX_pca);
    perf = classperf(testy,label);
    accuracy = [accuracy perf.CorrectRate];
    specificity = [specificity perf.Specificity];
    sensitivity = [sensitivity perf.Sensitivity];
    
%     yl=[];
%     for ee=1:size(train_AD_r,1)+newN_AD+size(train_MCI_r,1)+newN_MCI
%         if ee>=1 && ee<=size(train_AD_r,1)
%             yl=[yl;"real-AD"];
%         end
%         if ee>size(train_AD_r,1) && ee<=size(train_AD_r,1)+newN_AD
%             yl=[yl;"fake-AD"];
%         end
%         if ee>size(train_AD_r,1)+newN_AD && ee<=size(train_AD_r,1)+newN_AD+size(train_MCI_r,1)
%             yl=[yl;"real-MCI"];
%         end
%         if ee>size(train_AD_r,1)+newN_AD+size(train_MCI_r,1) && ee<=size(train_AD_r,1)+newN_AD+size(train_MCI_r,1)+newN_MCI
%             yl=[yl;"fake-MCI"];
%         end
%     end
%     nexttile
%     X_tsne = tsne(X);
%     colors = lines(6);
%     %yl = [ones(size(train_AD_r,1),1);ones(size(train_AD_fake,1),1)*2 ; ones(size(train_MCI_r,1),1)*3;ones(size(train_MCI_fake,1),1)*4];
%     gscatter(X_tsne(:,1),X_tsne(:,2),yl,colors,'.*')     
%     xlabel('pca1')
%     ylabel('pca2')
%     title(append('fold-',num2str(i)))

    
end
suptitle('Leftt-Hemisphere T-SNE of Real+Fake Samples')

%% graphs    
if drawGraph == 1
    %%% accuracy
    tiledlayout(2,2) % Requires R2019b or later
    nexttile
    b = bar(accuracy);
    ylim([0 1]);
    xlim([0 length(accuracy)+1]);
    xlabel('Models');
    ylabel('Accuracy');
    title('Model Accuracy');
    set(gca, 'XTick', 1:length(datasetNames),'XTickLabel',datasetNames);
    xtickangle(45)
    % first bar naming
    xtips1 = b(1).XEndPoints;
    ytips1 = b(1).YEndPoints;
    labels1 = string(b(1).YData);
    text(xtips1,ytips1,labels1,'VerticalAlignment','bottom',...
        'VerticalAlignment','bottom','rotation',45)

    %%% specificity
    nexttile
    b = bar(specificity);
    ylim([0 1]);
    xlim([0 length(specificity)+1]);
    xlabel('Models');
    ylabel('Specificity');
    title('Model Specificity');
    set(gca, 'XTick', 1:length(datasetNames),'XTickLabel',datasetNames);
    xtickangle(45)
    % first bar naming
    xtips1 = b(1).XEndPoints;
    ytips1 = b(1).YEndPoints;
    labels1 = string(b(1).YData);
    text(xtips1,ytips1,labels1,'VerticalAlignment','bottom',...
        'VerticalAlignment','bottom','rotation',45)


    %%% sensitivity
    nexttile
    b = bar(sensitivity);
    ylim([0 1]);
    xlim([0 length(sensitivity)+1]);
    xlabel('Models');
    ylabel('Sensitivity');
    title('Model Sensitivity');
    set(gca, 'XTick', 1:length(datasetNames),'XTickLabel',datasetNames);
    xtickangle(45)
    % first bar naming
    xtips1 = b(1).XEndPoints;
    ytips1 = b(1).YEndPoints;
    labels1 = string(b(1).YData);
    text(xtips1,ytips1,labels1,'VerticalAlignment','bottom',...
        'VerticalAlignment','bottom','rotation',45)

    %%% balanced accuracy
    nexttile
    b = bar((sensitivity+specificity)/2);
    ylim([0 1]);
    xlim([0 length(specificity)+1]);
    xlabel('Models');
    ylabel('Balanced Accuracy');
    title('Model Balanced Accuracy');
    set(gca, 'XTick', 1:length(datasetNames),'XTickLabel',datasetNames);
    xtickangle(45)
    % first bar naming
    xtips1 = b(1).XEndPoints;
    ytips1 = b(1).YEndPoints;
    labels1 = string(b(1).YData);
    text(xtips1,ytips1,labels1,'VerticalAlignment','bottom',...
        'VerticalAlignment','bottom','rotation',45)
end