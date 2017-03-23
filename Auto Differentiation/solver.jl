type Solver
   decayrate::Float64
   smootheps::Float64
   stepcache::Array{NNMatrix,1}
   Solver() = new(0.999, 1e-8, Array(NNMatrix,0))
end

function step_solver(solver::Solver, model::Model, stepsize::Float64, regc::Float64)
    # perform parameter update with RMSProp
    numtot = 0
    modelMatrices = model.matrices

    # init stepcache if needed
    if length(solver.stepcache) == 0
         for m in modelMatrices
            push!(solver.stepcache, NNMatrix(m.n, m.d))
        end
    end

    for k = 1:length(modelMatrices)
        @inbounds m = modelMatrices[k] # mat ref
        @inbounds s = solver.stepcache[k]
        for i in 1:m.n, j in 1:m.d
            # rmsprop adaptive learning rate
            @inbounds mdwi = m.dw[i,j]
            @inbounds s.w[i,j] = s.w[i,j] * solver.decayrate + (1.0 - solver.decayrate) * mdwi^2
            numtot += 1
            # update (and regularize)
            @inbounds m.w[i,j] += - stepsize * mdwi / sqrt(s.w[i,j] + solver.smootheps) - regc * m.w[i,j]
            @inbounds m.dw[i,j] = 0. # reset gradients for next iteration
        end
    end
end
