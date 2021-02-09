using DelimitedFiles;
dataset = readdlm("iris.data",',');
#Separamos los inputs de los outputs
inputs = dataset[:,1:4];
targets = dataset[:,5];
