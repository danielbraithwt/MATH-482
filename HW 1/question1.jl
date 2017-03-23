# Daniel Braithwaite
# 300313770

using Images
using DataFrames
using FileIO
using TestImages

ImageT = Array{ColorTypes.Gray{FixedPointNumbers.Normed{UInt8, 8}}, 2}
ArrT = Array{Float64, 2}

function compressImage(U, s, V, r)
  sc = zeros(length(s))
  for i in 1:r
    sc[i] = s[i]
  end

  n = U*diagm(sc)*V'
  return n
end

function compute2Norm(origonal, compressed)
  return norm(origonal - compressed)
end

function computeFrobNorm(origonal, compressed)
  return vecnorm(origonal - compressed)
end

compressionStops = [10, 20, 50, 100, 200, 400]
lena = testimage("lena_gray_512.tif")
lena_mat = convert(ArrT, lena)
U, s, V = svd(lena_mat)

origonal = U*diagm(s)*V'
orank = rank(lena_mat)

for stop in compressionStops
  compressed = compressImage(U, s, V, stop)

  sum = 0
  # sum = s[stop]
  for i in (stop+1):length(s)
    sum += s[i]^2
  end

  compressedRank = rank(compressed)
  error2 = compute2Norm(origonal, compressed)
  errorFrob = computeFrobNorm(origonal, compressed)

  println(string("Stop ", stop, ": rank=", compressedRank, " [origonal rank=", orank, "] error2=", error2, " errorFrob=", errorFrob, " sqrt(sum) = ", sqrt(sum)))
  Images.save(string("Julia_Rank", stop, ".png"), convert(ImageT, compressed))
end
