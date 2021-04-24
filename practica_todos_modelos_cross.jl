include("modulos/datasets.jl")
include("modulos/attributes_from_dataset.jl")
include("modulos/models_cross_validation.jl")

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
extraction = :A21;
dataset_name="datasets/faces.data";
change = false;
if extraction==:A1
    x = 6;
    y = 7;
elseif extraction==:A21
    x = 42;
    y = 43;
elseif extraction==:A22
    x = 36;
    y = 37;
end;
if (!isfile(dataset_name) || change)
    createDataset(dataset_name,extraction);
end;

dataset = readdlm(dataset_name,',');
inputs = convert(Array{Float64,2}, dataset[:,1:x]);
targets = convert(Array{Any,1},dataset[:,y]);
seed!(1);
numFolds = 10;

# Parametros principales de la RNA y del proceso de entrenamiento
topology = [9]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
validationRatio = 0.2; # Porcentaje de patrones que se usaran para validacion. Puede ser 0, para no usar validacion
maxEpochsVal = 30; # Numero de ciclos en los que si no se mejora el loss en el conjunto de validacion, se para el entrenamiento
numRepetitionsAANTraining = 50; # Numero de veces que se va a entrenar la RNA para cada fold por el hecho de ser no determinístico el entrenamiento
# Parametros del SVM
kernel = "rbf";
kernelDegree = 2;
kernelGamma = 15;
C=1;
# Parametros del arbol de decision
maxDepth = 4;
# Parapetros de kNN
numNeighbors = 5;
# Normalizamos las entradas, a pesar de que algunas se vayan a utilizar para test
#normalizeMinMax!(inputs);

# Entrenamos las RR.NN.AA.
modelHyperparameters = Dict();
modelHyperparameters["topology"] = topology;
modelHyperparameters["learningRate"] = learningRate;
modelHyperparameters["validationRatio"] = validationRatio;
modelHyperparameters["numExecutions"] = numRepetitionsAANTraining;
modelHyperparameters["maxEpochs"] = numMaxEpochs;
modelHyperparameters["maxEpochsVal"] = maxEpochsVal;
modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, numFolds);

# Entrenamos las SVM
modelHyperparameters = Dict();
modelHyperparameters["kernel"] = kernel;
modelHyperparameters["kernelDegree"] = kernelDegree;
modelHyperparameters["kernelGamma"] = kernelGamma;
modelHyperparameters["C"] = C;
modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, numFolds);

# Entrenamos los arboles de decision
modelCrossValidation(:DecisionTree, Dict("maxDepth" => maxDepth), inputs, targets, numFolds);

# Entrenamos los kNN
modelCrossValidation(:kNN, Dict("numNeighbors" => numNeighbors), inputs, targets, numFolds);
