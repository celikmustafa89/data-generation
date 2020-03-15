function [mat] = unVectorizeSigma(vector)

n = round((sqrt(8 * numel(vector) + 1) - 1) / 2);
M = zeros(n, n);
c = 0;
for i2 = 1:n
  for i1 = 1:i2
    c         = c + 1;
    M(i1, i2) = vector(c);
  end
end

[n,m] = size(M);
mat = M'+M;
mat(1:n+1:end) = diag(M);