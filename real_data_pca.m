function [accuracy,specificity,sensitivity] = real_data_pca(X_AD_tensors,X_MCI_tensors, drawGraph)

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
for i=1:fold_k
    datasetNames{i} = append('fold-',num2str(i));
    
    % divides test set and train set with fold manner
    [datasetTrain_AD,datasetTest_AD,datasetTrain_MCI,datasetTest_MCI] = get_fold(X_AD_tensor_folds,X_MCI_tensor_folds,fold_k,i);
   
    % vectorize features for train samples
    train_AD_r = cell2vec(datasetTrain_AD);
    train_MCI_r = cell2vec(datasetTrain_MCI);
    % vectorize features for test samples
    test_AD_r = cell2vec(datasetTest_AD);
    test_MCI_r = cell2vec(datasetTest_MCI);
    % combine two different populations(AD and MCI) in same train array
    X = [train_AD_r ; train_MCI_r];
    y = [ones(size(train_AD_r,1),1) ; zeros(size(train_MCI_r,1),1)];
    % combine two different populations(AD and MCI) in same test array
    testX = [test_AD_r ; test_MCI_r];
    testy = [ones(size(test_AD_r,1),1) ; zeros(size(test_MCI_r,1),1)];
    

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
    
end

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