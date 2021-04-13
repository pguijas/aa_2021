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
@sk_import svm: SVC
using Random
using Random:seed!

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


# Creamos nuestra máquina de soporte vectorial
# @ arguments
#   -kernel: función de kernel que usará la svm ‘linear’, ‘poly’, ‘rbf’,
#       ‘sigmoid’, ‘precomputed’ o una función que definamos nosotros.
#   -degree:
model = SVC(kernel="rbf", degree=3, gamma=2, C=1);

model = fit!(model, trainingInputs, trainingTargets);

distances = decision_function(model, inputs);

@show(distances)
@show(keys(model))
@show(model.C)
@show(model.support_vectors_)
@show(model.support_)