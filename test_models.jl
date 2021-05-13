include("modulos/testing_models.jl")
include("modulos/datasets.jl")
include("modulos/attributes_from_dataset.jl")

#=
hay diferentes extracciones:
    · :A1 (aproximación 1)
    · :A21 (aproximación 2 extracción 1)
    · :A22 (aproximación 2 extracción 2)
    etc...
=#
extraction = :A33;
dataset_name="datasets/faces.data";
change = false;
if extraction==:A1
    x = 6;
    y = 7;
elseif extraction==:A21
    x = 42;
    y = 43;
elseif (extraction==:A22 || extraction==:A23)
    x = 36;
    y = 37;
elseif extraction==:A31
    x = 48;
    y = 49;
elseif extraction==:A32
    x = 66;
    y = 67;
elseif extraction==:A33
    x = 72;
    y = 73;
elseif extraction==:A34
    x = 84;
    y = 85;
end;
if (!isfile(dataset_name) || change)
    createDataset(dataset_name,extraction);
end;

dataset = readdlm(dataset_name,',');
inputs = convert(Array{Float64,2}, dataset[:,1:x]);
targets = convert(Array{Any,1},dataset[:,y]);
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


=#
# Entrenamos knn
testingModels(:KNN, Dict("maxNeighbors" => 20), inputs, targets, numFolds; rep=:All);

# Entrenamos los arboles de decision
testingModels(:DecisionTree, Dict("maxDepth" => 20), inputs, targets, numFolds; rep=:All);

# Entrenamos svm
modelHyperparameters = Dict();
modelHyperparameters["kernel"] = "rbf";
modelHyperparameters["kernelDegree"] = 15;
modelHyperparameters["maxGamma"] = 10;
testingModels(:SVM, modelHyperparameters, inputs, targets, numFolds; rep=:All);
#=
# Entrenamos las RR.NN.AA.
modelHyperparameters = Dict();
modelHyperparameters["fstNeuron"] = 1;
modelHyperparameters["maxNNxlayer"] = 16;
modelHyperparameters["layers"] = 1;
testingModels(:ANN, modelHyperparameters, inputs, targets, numFolds; rep=:All);


modelHyperparameters = Dict();
modelHyperparameters["fstNeuron"] = 6;
modelHyperparameters["maxNNxlayer"] = 12;
modelHyperparameters["layers"] = 2;
testingModels(:ANN, modelHyperparameters, inputs, targets, numFolds; rep=:All);
=#
