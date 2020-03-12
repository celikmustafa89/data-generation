function [child1, child2]=crossoverAndMutate(parent1, parent2,index,crossover_type, spd_type)

v1 = vectorizeSigma(parent1);
v2 = vectorizeSigma(parent2);

% crossover method1
% parent1 = [1 1 1 1 1 1 1 1];
% parent2 = [2 2 2 2 2 2 2 2];
% child1 =  [1 2 1 2 1 2 1 2];
% child2 =  [2 1 2 1 2 1 2 1];
if strcmp(crossover_type,'mix')
    v1_child = v1;
    v2_child = v2;
    for i=1:2:length(v1)
        v1_child(i) = v2(i);
        v2_child(i) = v1(i);
    end
end

% crossover method2
% parent1 = [1 1 1 1 1 1 1 1];
% parent2 = [2 2 2 2 2 2 2 2];
% child1 =  [1 1 1 1 2 2 2 2];
% child2 =  [2 2 2 2 1 1 1 1];
if strcmp(crossover_type,'half')
    v1_child = [v1(1:index) v2(index+1:end)];
    v2_child = [v2(1:index) v1(index+1:end)];
end

%% mutation

v1_child(1) = v1_child(1)+randi([-1 1])*(v1_child(1)/4);
v2_child(end) = v2_child(end)+randi([-1 1])*(v2_child(end)/4);
v1_child(2) = v1_child(2)+randi([-1 1])*(v1_child(2)/4);
v2_child(end-1) = v2_child(end-1)+randi([-1 1])*(v2_child(end-1)/4);


v1_child_matrix = unVectorizeSigma(v1_child);
v2_child_matrix = unVectorizeSigma(v2_child);

%% nearest symmetric possitive defiente matrix conversion version 1
if strcmp(spd_type,'version1')
    child1 = nearest_posdef(v1_child_matrix);
    child2 = nearest_posdef(v2_child_matrix);
end

%% nearest symmetric possitive defiente matrix conversion version 2
if strcmp(spd_type,'version2')
    child1 = nearestSPD(v1_child_matrix);
    child2 = nearestSPD(v2_child_matrix);
end


