include("modulos/datasets.jl")
include("modulos/attributes_from_dataset.jl")
include("modulos/models_cross_validation.jl")


#=
hay tres extracciones
    · :A1 (aproximación 1)
    · :A21 (aproximación 2 extracción 1)
    · :A22 (aproximación 2 extracción 2)
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

# Parametros principales de la RNA y del proceso de entrenamiento
topology = [3];
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
validationRatio = 0.2; # Porcentaje de patrones que se usaran para validacion. Puede ser 0, para no usar validacion
maxEpochsVal = 30; # Numero de ciclos en los que si no se mejora el loss en el conjunto de validacion, se para el entrenamiento
numRepetitionsAANTraining = 50; # Numero de veces que se va a entrenar la RNA para cada fold por el hecho de ser no determinístico el entrenamiento
# Parametros del SVM
kernel = "rbf";
kernelDegree = 2;
kernelGamma = 1;
C=1000000;
# Parametros del arbol de decision
maxDepth = 15;
# Parapetros de kNN
numNeighbors = 1;
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
