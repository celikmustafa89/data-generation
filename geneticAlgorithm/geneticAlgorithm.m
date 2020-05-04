function [best_sigmas, best_sigma_dists, best_view] = geneticAlgorithm(eval_view,N,nv,numROIs,representative_tensor,NumFeatures,crossover_type,spd_type,generation_type)


%0. initialization of the variables
%N = LH_train_N;
%nv = LH_train_nv;
%numROIs = LH_train_numROIs;
%eval_view = LH_eval_view;
%representative_tensor = LH_train_representative_tensor;
%NumFeatures = LH_eval_Numfeatures;

%1. generate sigma value as population
% burada random olarak 60 tane sigma value olusturuluyor.
% sonrasında bu sigma value ile genetik algoritması calıstırılacak
numPopulation = 60;
sigmas = {numPopulation}; 

for i=1:numPopulation
    rng(i)
    ii = rand(nv);
    sigmas{i} = ii*ii.';
end
%%%
fprintf(' %i random sigma values are generated.\n', numPopulation)

m=1; % best sigma counter
itr=0;
%for itr=1:numIteration
while 1
    itr=itr+1;
    fprintf('iteration: %i\n', itr)
    sigma_distance = zeros(length(sigmas));
    newViews = {};
    for i=1:length(sigmas)
        % 2. generate samples by using sigma values 
        % and 
        % 3. calculate cross-distance(fitness) for each sigma
        % burada sigma value'lar kullanılarak sample'lar üretiliyor ve
        % sonrasında cross-distance hesabı yapılarak her bir sigmanın cross
        % distance degeri bulunuyor.
        [sigma_distance(i) newViews{i}] = fitness_function(eval_view,representative_tensor,N,nv,NumFeatures,numROIs,sigmas{i},generation_type);
    end
    

    % 4. sort sigma value by looking fitnesses
    % [out,idx] = sort([14 8 91 19])
    % burada sigmalara karsılık bulunan cross-distance degerleri
    % sıralanıyor.
    [~,sorted_ids] = sort(sigma_distance);

    % burası 1 iterationda bir en iyi sonucu alıyor
    %if rem(itr,1) == 0
        best_sigma_dists(m) = sigma_distance(sorted_ids(1));
        best_view = newViews{sorted_ids(1)};
        fprintf('sigma distance: %i\n', best_sigma_dists(m));
        m=m+1;
    %end

    % 5. select   sigma values as parents
    % sıralanan degerler arasında en basarılı sigma degerleri seçilecek
    % 5.1 decide number of parent for crossover
    % burada kaç tane parent secilmesi gerektiği bulunuyor.

    for i=1:numPopulation
        if i*(i+1) > numPopulation
            numMatingParents = i;
            break;
        end
    end
    
    % 5.2 select best sigmas as parent subjects
    % burada en iyi parent'lar seciliyor
    parent_sigmas = {numPopulation};
    for i=1:numMatingParents+1
        parent_sigmas{i} = sigmas{sorted_ids(i)};
    end

    % 6 make crossover and mutation
    % 6.1 do crossover for each parent in the parent population
    
    split_index = ((size(parent_sigmas{1},1) *(size(parent_sigmas{1},1)-1))/2) + size(parent_sigmas{1},1);
    split_index = split_index/2;
    
    %for t=2:((size(parent_sigmas{1},1) *(size(parent_sigmas{1},1)-1))/2) + size(parent_sigmas{1},1)-1
     %   if t == split_index
     %       continue;
     %   end
    k=1;
    for i=1:numMatingParents-1
        for j=i+1:numMatingParents
            [sigma1,sigma2] = crossoverAndMutate(parent_sigmas{i},parent_sigmas{j},split_index,crossover_type,spd_type);
            new_sigmas{k} = sigma1;
            k = k+1;
            new_sigmas{k} = sigma2;
            k = k+1;
        end
    end
    
    %end
    %6.2 generate missing subjects for population
%     for i=1:(numPopulation-numMatingParents*(numMatingParents-1))/2
%         [sigma1,sigma2] = crossoverAndMutate(parent_sigmas{numMatingParents+1},parent_sigmas{i},split_index);
%         sigmas{k} = sigma1;
%         k=k+1;
%         sigmas{k} = sigma2;
%         k=k+1;
%     end

    % 6.2.1 combine new and old sigma value
    for i=1:numPopulation
        new_sigmas{k} = sigmas{sorted_ids(i)};
        k = k+1;
    end
    sigmas = new_sigmas;
    
    % 7. plot the graph
    
    b = bar(best_sigma_dists)
    ylim([0 40]);
    xlim([0 m+1]);
    xlabel('Dataset Name');
    ylabel('Distance between Evaluation and Fake Samples');
    title('Cross Distance Between Fake and Evaluation Dataset');
    set(gca, 'XTick', 1:m,'XTickLabel',[1:1:m]);
    xtickangle(45)
   
    xtips1 = b(1).XEndPoints;
    ytips1 = b(1).YEndPoints;
    labels1 = string(b(1).YData);
    text(xtips1,ytips1,labels1,'VerticalAlignment','bottom',...
        'VerticalAlignment','bottom','rotation',75)

    if itr == 5
        break;
    end
end



sigma_distance = zeros(length(sigmas));
for i=1:length(sigmas)
    %2. generate samples by using sigma values
    %[samples,fakeView] = generateSampleGA(LH_train_representative_tensor,N,nv,NumFeatures,numROIs, sigmas{i});

    %3. calculate cross-distance(fitness) for each sigma
    [sigma_distance(i) newViews{i}] =fitness_function(eval_view,representative_tensor,N,nv,NumFeatures,numROIs,sigmas{i},generation_type);
end

%4. sort sigma value by looking fitnesses
%[out,idx] = sort([14 8 91 19])
[~,sorted_ids] = sort(sigma_distance);
best_sigmas = {numPopulation};
for i=1:numPopulation
    best_sigmas{i} = sigmas{sorted_ids(i)};
    sorted_sigma_distances(i) = sigma_distance(sorted_ids(i));
end
