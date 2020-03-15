function [total_dist]=cross_distance(view1,N1,view2,N2,nv)

total_dist = 0;
for i=1:N1
    second_view_total = 0;
    for j=1:N2
        Frob_dist=0;
        for k=1:nv
            Frob_dist = Frob_dist + norm(view1{k}(:,:,i)-view2{k}(:,:,j),'fro');
        end
        second_view_total = second_view_total + Frob_dist/nv;
    end
    total_dist = total_dist + second_view_total/N2;
end
total_dist=total_dist/N1;