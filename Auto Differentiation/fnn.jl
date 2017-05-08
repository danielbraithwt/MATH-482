type FNNLayer
    wxh::NNMatrix
    bhh::NNMatrix
    function FNNLayer(prevsize::Int, hiddensize::Int, std::Float64)
        wxh = randNNMat(hiddensize, prevsize, std)
        wbh = NNMatrix(hiddensize, 1, zeros(hiddensize,1), zeros(hiddensize,1))
        new(wxh, wbh)
    end
end

type FNN <: Model
    hdlayers::Array{FNNLayer,1}
    whd::NNMatrix
    bd::NNMatrix
    matrices::Array{NNMatrix,1}
    hiddensizes::Array{Int,1}
    function FNN(inputsize::Int, hiddensizes::Array{Int,1}, outputsize::Int, std::Float64=0.08)
        hdlayers = Array(FNNLayer, length(hiddensizes))
        matrices = Array(NNMatrix, 0)
        for d in 1:length(hiddensizes)
            prevsize = d == 1 ? inputsize : hiddensizes[d-1]
            layer = FNNLayer(prevsize, hiddensizes[d], std)
            hdlayers[d] = layer
            push!(matrices, layer.wxh)
            push!(matrices, layer.bhh)
        end
        whd = randNNMat(outputsize, hiddensizes[end], std)
        bd = NNMatrix(outputsize, 1, zeros(outputsize,1), zeros(outputsize,1))
        push!(matrices, whd)
        push!(matrices, bd)
        new(hdlayers, whd, bd, matrices, hiddensizes)
    end
end

function forwardprop(g::Graph, model::FNN, x)
    hidden = Array(NNMatrix,0)
    # Move through all the hidden layers
    for d in 1:length(model.hiddensizes)
        # If we are at the first layer then use the input vector as input
        # otherwise we use the output of the previous layer
        input = d == 1 ? x : hidden[d-1]

        # Get the weights and byas for this layer
        wxh = model.hdlayers[d].wxh
        bhh = model.hdlayers[d].bhh

        # Compute the apply the weights to the input
        h0 = mul(g, wxh, input)

        # Add the byas to the weight array of (inputs * weights)
        # Finally apply the tanh function (the activation function) to the
        hidden_d = relu(g, add(g, h0, bhh))

        # Add the output of this layer to the list of outputs
        push!(hidden,hidden_d)
    end

    # After we have applyed the hidden layers, apply the hidden layer weights
    # and bays
    output = add(g, mul(g, model.whd, hidden[end]), model.bd)
    return output
end
