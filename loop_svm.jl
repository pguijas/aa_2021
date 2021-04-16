include("modulos/datasets.jl")
include("modulos/attributes_from_dataset.jl")
include("modulos/models_cross_validation.jl")
using Flux

seed!(1);

dataset_name="datasets/faces.data"
if (!isfile(dataset_name))
    (inputs, targets) = getInputs("../AA_DATASET");
    println("Tamaños en la generación:")
    println(size(inputs))
    println(size(targets))
    write_dataset(dataset_name,inputs,targets)
end

dataset = readdlm(dataset_name,',');
inputs = convert(Array{Float64,2}, dataset[:,1:6]);             #Array{Float64,2}
targets = convert(Array{Any,1},dataset[:,7]);                   #Array{Bool,2}


numFolds = 10;

# Parametros del SVM
kernel = "rbf"; #linear, poly, rbf, sigmoid, precomputed
kernelDegree = 3; # en caso de que sea 'poly', indica el grado del polinomio (num imputs)
kernelGamma = 2; #coeficiente del kernel para rbf, poly y sigmoid
C=1; #parámetro de regularización

# Entrenamos las SVM
modelHyperparameters = Dict();
modelHyperparameters["kernel"] = kernel;
modelHyperparameters["kernelDegree"] = kernelDegree;
modelHyperparameters["kernelGamma"] = kernelGamma;
modelHyperparameters["C"] = C;
modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, numFolds);
