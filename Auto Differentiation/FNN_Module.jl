# push!(LOAD_PATH, pwd())
module FNN_Module
    export Model, FNN 
    export NNMatrix, randNNMat, forwardprop, Solver, step_solver
    export Graph, backprop

    include("nnmatrix.jl")
    include("graph.jl")
    include("solver.jl")
    include("fnn.jl")
end 
