using DelimitedFiles;
dataset = readdlm("iris.data",',');
#Separamos los inputs de los outputs
inputs = dataset[:,1:4];
targets = dataset[:,5];
#Debemos converertir los tipos (concretar) Any -> Float64
inputs = convert(Array{Float64,2},inputs);
targets = convert(Array{Float64,2},targets);
