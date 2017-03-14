B = pinv(X'*X)*(X'*Y)
println(string("B = ", B))

YFitted = zeros(n)
for i in 1:n
  YFitted[i] = B[1] * X[i]
end

plot(X, YFitted)
