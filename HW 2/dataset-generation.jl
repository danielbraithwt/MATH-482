# Daniel Braithwaite
# 300313770
using PyPlot

n = 100
xMax = 100
b = 0.5
errorMax = 15

srand(1234)

X = rand(1:xMax, n)
Y = zeros(n)

for i in 1:n
  Y[i] = b * X[i] + rand(1:errorMax)
end

plot(X, Y, "ro", label="Data Set")
