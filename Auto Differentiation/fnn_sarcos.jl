push!(LOAD_PATH, pwd())
using FNN_Module
using MAT

function normalize(x)
    m, n = size(x)
    y = x .- sum(x,2) / n
    y = y ./ sqrt(sum(y.^2,2) / n)
    return y
end

function load_matlab_data(filename::String, dictname::String)
    x = matread(filename)[dictname]
    return normalize(x)
end

function test_sarcos(model::Model, test_points::Array, test_labels::Array)
    g = Graph(false)
    n, d = size(test_points)
    predictions = zeros((n, 7, 1))
    for idx in 1:n
        x = NNMatrix(d, 1, test_points[idx,:]'')
        pred = forwardprop(g, model, x)
        predictions[idx, :, :] = pred.w
    end
    return mean((predictions - test_labels).^2)
end

function cost_func(model::Model, query_points::Array, query_labels::Array, mb_size::Int64)
    n, d = size(query_points)
    sample = rand(1:n, mb_size)
    g =  Graph()
    cost = 0.0
    for idx in 1:mb_size
        x = NNMatrix(d,1,query_points[sample[idx],:]'')
        pred = forwardprop(g, model, x)
        cost += (pred.w - query_labels[sample[idx]]).^2
        pred.dw = (pred.w - query_labels[sample[idx]])
    end
    return g, sum(cost)
end

function train_model(model::Model, query_points::Array, query_labels::Array, solver::Solver)
    maxIter = 10000
    mb_size = 32
    tickiter = 1
    regc = 0.000001
    learning_rate = 0.001

    while tickiter < maxIter
        g, cost = cost_func(model, query_points, query_labels, mb_size)
        backprop(g)
        step_solver(solver, model, learning_rate, regc)
        tickiter += 1
        if (tickiter % 1000) == 0
            println("Loss = $(round(cost,2)) @ $tickiter")
        end
    end
    return model, solver
end


srand(12345)

sarcos_train = load_matlab_data("sarcos_inv.mat", "sarcos_inv")
sarcos_test = load_matlab_data("sarcos_inv_test.mat", "sarcos_inv_test") 
query_points = sarcos_train[:,1:21]
query_labels = sarcos_train[:,22:28]
test_points = sarcos_test[:,1:21]
test_labels = sarcos_test[:,22:28]

solver = Solver() # RMSProp optimizer
model = FNN(21, [301, 101], 7)

model, solver = train_model(model, query_points, query_labels, solver)
println("Test performance = $(round(test_sarcos(model, test_points, test_labels),2))")
