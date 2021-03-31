# antes de hacer esto importante descargarlo para python
# pip3 install -U scikit-learn
# Reiniciamos Julia con CTR+D + ENTER
# SI ejecutamos el archivo debería ir todo bien

#import Pkg;
#Pkg.add("ScikitLearn")
#Pkg.update()
using FileIO;
using DelimitedFiles;
using ScikitLearn;
@sk_import neighbors: KNeighborsClassifier

include("../modulos/datasets.jl")

# Cogemos los inputs y los dividimos
dataset = readdlm("../datasets/iris.data",',');
inputs = convert(Array{Float64,2}, dataset[:,1:4]);
targets = dataset[:,5];
normalizeMinMax!(inputs);
(trainingIndices, _, testIndices) = holdOut(size(inputs,1), 0.2, 0.2);

trainingInputs = inputs[trainingIndices,:];
testInputs = inputs[testIndices,:];
trainingTargets = targets[trainingIndices,:];
testTargets = targets[testIndices,:];


model = KNeighborsClassifier(3);

model = fit!(model, trainingInputs, trainingTargets);

@show(keys(model))
@show(model.n_neighbors)
@show(model.metric)
@show(model.weights)
