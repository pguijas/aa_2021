include("modulos/testing_models.jl")


dataset_name="datasets/faces.data"
if (!isfile(dataset_name))
    (inputs, targets) = getInputs("datasets");
    println("Tamaños en la generación:")
    println(size(inputs))
    println(size(targets))
    write_dataset(dataset_name,inputs,targets)
end

dataset = readdlm(dataset_name,',');

#inputs = convert(Array{Float64,2}, dataset[:,1:6]);
#targets = convert(Array{Any,1},dataset[:,7]);

inputs = convert(Array{Float64,2}, dataset[:,1:42]);
targets = convert(Array{Any,1},dataset[:,43]);

seed!(1);

numFolds = 10;

# Entrenamos knn
testingModels(:KNN, Dict("maxNeighbors" => 40), inputs, targets, numFolds, :AccStd);

# Entrenamos los arboles de decision
testingModels(:DecisionTree, Dict("maxDepth" => 40), inputs, targets, numFolds, :AccStd);
# Entrenamos svm
modelHyperparameters = Dict();
modelHyperparameters["kernel"] = "rbf";
modelHyperparameters["kernelDegree"] = 1;
modelHyperparameters["maxGamma"] = 20;
testingModels(:SVM, modelHyperparameters, inputs, targets, numFolds, :AccStd);
#=
# Entrenamos las RR.NN.AA.
#modelHyperparameters = Dict();
#modelHyperparameters["fstNeuron"] = 1;
#modelHyperparameters["maxNNxlayer"] = 2;
#modelHyperparameters["layers"] = 1;
#testingModels(:ANN, modelHyperparameters, inputs, targets, numFolds, :AccStd);
=#
