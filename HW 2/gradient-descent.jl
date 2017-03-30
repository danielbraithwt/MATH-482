# Daniel Braithwaite
# 300313770

learningRate = 0.025
w = 0
goalError = 0.001

function err(w)
  sum = 0
  for i in 1:length(X)
    sum += (w*X[i] - Y[i])^2
  end

  return 1/(length(X)) * sum
end

function grad(w)
  sum = 0
  for i in 1:length(X)
    sum += X[i] * (w*X[i] - Y[i])
  end

  return 2/(length(X)) * sum
end

t = 1
while err(w) > goalError && t < 500
  g = grad(w)
  w = w - (1/t) * learningRate * g
  t += 1
end

println("\nGradient Decent: ")
println(string("W = ", w))
println(string("Error = ", err(w)))

YFitted = zeros(n)
for i in 1:n
  YFitted[i] = w * X[i]
end

plot(X, YFitted, color="g")
