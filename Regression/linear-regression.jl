function err(w)
  sum = 0
  for i in 1:length(X)
    sum += (w*X[i] - Y[i])^2
  end

  return 1/(length(X)) * sum
end

B = pinv(X'*X)*(X'*Y)
println("\nLinear Regression: ")
println(string("B = ", B[1]))
println(string("Error = ", err(B[1])))


YFitted = zeros(n)
for i in 1:n
  YFitted[i] = B[1] * X[i]
end

plot(X, YFitted)
