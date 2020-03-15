function [Frob_dist]=frobenious(view,CBT,nv)

Frob_dist=0;
for k=1:nv
    Frob_dist=Frob_dist+FrobMetric(view{k},CBT);
end
Frob_dist=Frob_dist/nv;