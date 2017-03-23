import Base.tanh

type Graph
   backprop::Array{Function,1}
   doBackprop::Bool 
   Graph() = new(Array(Function,0),true)
   Graph(backPropNeeded::Bool) = new(Array(Function,0),backPropNeeded)
end

function backprop(g::Graph)
    for i = length(g.backprop):-1:1  g.backprop[i]() end
end

function tanh(g::Graph, m::NNMatrix)
    out = NNMatrix(m.n, m.d)
    out.w = tanh(m.w)
    if g.doBackprop
        push!(g.backprop,
              function ()
                  @inbounds for j in 1:m.d, i in 1:m.n
                      m.dw[i,j] += (1. - out.w[i,j]^2) * out.dw[i,j]
                  end
              end )
    end
    return out
end

function mul(g::Graph, m1::NNMatrix, m2::NNMatrix)
    out = NNMatrix(m1.n, m2.d, m1.w * m2.w, zeros(m1.n, m2.d))
    if g.doBackprop
        push!(g.backprop,
            function ()
                @inbounds for i in 1:m1.n, j in 1:m2.d
                    b = out.dw[i,j]
                    for k in 1:m1.d
                        m1.dw[i,k] += m2.w[k,j] * b

                        m2.dw[k,j] += m1.w[i,k] * b
                    end
                end
            end )
    end
    return out
end

function mul(g::Graph, m1::NNMatrix, m2::Array{Float64})
    out = NNMatrix(m1.n, 1, m1.w * m2, zeros(m1.n, 1))
    if g.doBackprop
        push!(g.backprop,
            function ()
                @inbounds for i in 1:m1.n
                    b = out.dw[i,1]
                    for k in 1:m1.d
                        m1.dw[i,k] += m2[k,1] * b
                    end
                end
            end )
    end
    return out
end


function mul(g::Graph, m::NNMatrix, c::Float64)
    out = NNMatrix(m.n, m.d, m.w .* c, zeros(m.n, m.d))
    if g.doBackprop
        push!(g.backprop,
            function ()
                m.dw += out.dw .* c
            end )
    end
    return out
end

function sub(g::Graph, m1::NNMatrix, m2::NNMatrix)
    out = NNMatrix(m1.n, m1.d, m1.w - m2.w, zeros(m1.n, m1.d))
    if g.doBackprop
        push!(g.backprop,
            function ()
                m1.dw += out.dw
                m2.dw -= out.dw                
            end )
    end
    return out
end

function add(g::Graph, ms::NNMatrix...)
    out = NNMatrix(ms[1].n, ms[1].d, zeros(ms[1].n, ms[1].d), zeros(ms[1].n, ms[1].d))
    @inbounds for m in ms
        @inbounds for j in 1:m.d, i in 1:m.n
            out.w[i,j] += m.w[i,j]
        end
    end
    if g.doBackprop
        push!(g.backprop,
            function ()
                @inbounds for m in ms
                    @inbounds for j in 1:m.d, i in 1:m.n
                        m.dw[i,j] += out.dw[i,j]
                    end
                end
            end )
    end
    return out
end

function add(g::Graph, m::NNMatrix, c::Float64)
    out = NNMatrix(m.n, m.d, m.w .+ c, zeros(m.n, m.d))
    if g.doBackprop
        push!(g.backprop,
            function ()
                m.dw += out.dw
            end )
    end
    return out
end

