using DelimitedFiles;
dataset = readdlm("iris.data",',');
#Separamos los inputs de los outputs
inputs = dataset[:,1:4];
targets = dataset[:,5];
#Debemos converertir los tipos (concretar) Any -> Float64
inputs = convert(Array{Float64,2},inputs);
#Con los targets adaptar el problema (3 categorías en este caso)
#   Iris Virginica
#   Iris Versicolour
#   Iris Setosa
targets = convert(Vector{String},targets); #supongo que hay que transformarlo previamente a string para tratarlo como tal
# [IrisVi IrisVe IrisS] bool array
normalizador_chulisimo(x::String) = [x=="Iris-virginica" x=="Iris-versicolor" x=="Iris-setosa"]
targets = normalizador_chulisimo.(targets)
