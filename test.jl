include("modulos/testing_models.jl")
include("modulos/datasets.jl")
include("modulos/attributes_from_dataset.jl")

function createDataset(dataset_name::String, extraction::Symbol)
    (inputs, targets) = getInputs("datasets"; extr=extraction);
    println("Tamaños en la generación:");
    println(size(inputs));
    println(size(targets));
    write_dataset(dataset_name,inputs,targets);
    while (!isfile("datasets/faces.data"))
        sleep(1);
    end;
end;

#=
hay tres extracciones
    · :A1 (aproximación 1)
    · :A21 (aproximación 2 extracción 1)
    · :A22 (aproximación 2 extracción 2)
=#
extraction = :A31;
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


# Entrenamos knn
testingModels(:KNN, Dict("maxNeighbors" => 40), inputs, targets, numFolds; rep=:All);

# Entrenamos los arboles de decision
testingModels(:DecisionTree, Dict("maxDepth" => 40), inputs, targets, numFolds; rep=:All);
=#

# Entrenamos svm
modelHyperparameters = Dict();
modelHyperparameters["kernel"] = "rbf";
modelHyperparameters["kernelDegree"] = 15;
modelHyperparameters["maxGamma"] = 20;
testingModels(:SVM, modelHyperparameters, inputs, targets, numFolds; rep=:All);

#=
# Entrenamos las RR.NN.AA.
modelHyperparameters = Dict();
modelHyperparameters["fstNeuron"] = 1;
modelHyperparameters["maxNNxlayer"] = 16;
modelHyperparameters["layers"] = 1;
testingModels(:ANN, modelHyperparameters, inputs, targets, numFolds; rep=:All);
=#

#=
modelHyperparameters = Dict();
modelHyperparameters["fstNeuron"] = 1;
modelHyperparameters["maxNNxlayer"] = 8;
modelHyperparameters["layers"] = 2;
testingModels(:ANN, modelHyperparameters, inputs, targets, numFolds; rep=:All);
=#
