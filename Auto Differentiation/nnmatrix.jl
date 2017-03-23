abstract Model 

type NNMatrix 
    n::Int
    d::Int
    w::Matrix{Float64}
    dw::Matrix{Float64}
    NNMatrix(n::Int) = new(n, 1, zeros(n,1), zeros(n,1))
    NNMatrix(n::Int, d::Int) = new(n, d, zeros(n,d), zeros(n,d))
    NNMatrix(n::Int, d::Int, w::Array) = new(n, d, w, zeros(n,d))
    NNMatrix(n::Int, d::Int, w::Array, dw::Array) = new(n, d, w, dw)
end

randNNMat(n::Int, d::Int, std::AbstractFloat=1.) = NNMatrix(n, d, randn(n,d)*std, zeros(n,d))
