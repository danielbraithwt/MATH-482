# Daniel Braithwaite
# 300313770

using FileIO, Images

ImageT = Array{ColorTypes.Gray{FixedPointNumbers.Normed{UInt8, 8}}, 2}
ArrT = Array{Float64, 2}

x = 243; y = 320; l = x*y

dir = "./data/"

function load_faces(dir, l)
  images = readdir(dir)[2:end]
  n = length(images)
  X = zeros((n,l))
  for i in 1:n
    im = convert(ArrT, convert(ImageT, load(dir * images[i])))
    X[i,:] = reshape(im, 243*320)
  end

  return X
end

function save_face(data, dir, name)
  img = reshape(data, (243, 320))
  Images.save(string(dir, name, ".png"), convert(ImageT, img))
end

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

facesToSave = [5, 10, 14, 50, 67, 100]
compressionStops = [10, 20, 40, 80, 120]
faces = load_faces(dir, l)
U, s, V = svd(faces)
origonal = U*diagm(s)*V'
orank = rank(origonal)

for stop in compressionStops
  compressed = compressImage(U, s, V, stop)

  compressedRank = rank(compressed)
  error2 = compute2Norm(origonal, compressed)
  errorFrob = computeFrobNorm(origonal, compressed)

  # Rounding errors causing values out of range
  for i in 1:length(compressed)
    if compressed[i] > 1
      compressed[i] = 1
    end

    if compressed[i] < 0
      compressed[i] = 0
    end
  end

  println(string("Stop ", stop, ": rank=", compressedRank, " [origonal rank=", orank, "] error2=", error2, " errorFrob=", errorFrob))

  for f in facesToSave
    save_face(compressed[f,:], string(""), string("Image-", f, "-", stop))
  end
end
