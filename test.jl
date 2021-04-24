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
inputs = convert(Array{Float64,2}, dataset[:,1:42]);
targets = convert(Array{Any,1},dataset[:,43]);
seed!(1);
numFolds = 10;


#=
Posibles representaciones
    · ninguna o :All = saca todo por pantalla.
    · :AccStd = accuracy y su std.
    · :F1Std = F1 y su std.
    · :AccF1 = accuracy y F1.
    · :Plot3D = solo para svm con kernel polinómico.

Para sacar los gráficos 3D:
sudo apt-get install python3-matplotlib
Pkg.add("PyPlot") || Pkg.build("PyPlot")


# Entrenamos knn
testingModels(:KNN, Dict("maxNeighbors" => 40), inputs, targets, numFolds; rep=:All);

# Entrenamos los arboles de decision
testingModels(:DecisionTree, Dict("maxDepth" => 40), inputs, targets, numFolds; rep=:All);
=#

#=
# Entrenamos svm
modelHyperparameters = Dict();
modelHyperparameters["kernel"] = "poly";
modelHyperparameters["kernelDegree"] = 15;
modelHyperparameters["maxGamma"] = 20;
testingModels(:SVM, modelHyperparameters, inputs, targets, numFolds; rep=:Plot3D);

# Entrenamos las RR.NN.AA.
modelHyperparameters = Dict();
modelHyperparameters["fstNeuron"] = 1;
modelHyperparameters["maxNNxlayer"] = 16;
modelHyperparameters["layers"] = 1;
testingModels(:ANN, modelHyperparameters, inputs, targets, numFolds; rep=:All);
=#

modelHyperparameters = Dict();
modelHyperparameters["fstNeuron"] = 1;
modelHyperparameters["maxNNxlayer"] = 8;
modelHyperparameters["layers"] = 2;
testingModels(:ANN, modelHyperparameters, inputs, targets, numFolds; rep=:All);
#=
=#
