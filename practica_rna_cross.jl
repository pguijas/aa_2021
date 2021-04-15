include("modulos/bondad.jl")
include("modulos/graphics.jl")
include("modulos/datasets.jl")
include("modulos/rna.jl")
include("modulos/attributes_from_dataset.jl")
include("modulos/models_cross_validation.jl")

using Random
using Random:seed!
seed!(33);

#nº elementos

# Parametros principales de la RNA y del proceso de entrenamiento
topology = [4, 3];              # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01;            # Tasa de aprendizaje
numMaxEpochs = 1000;            # Numero maximo de ciclos de entrenamiento
numFolds = 10;
validationRatio = 0.2;          # Porcentaje de patrones que se usaran para validacion. Puede ser 0, para no usar validacion
maxEpochsVal = 6;               # Numero de ciclos en los que si no se mejora el loss en el conjunto de validacion, se para el entrenamiento
numRepetitionsAANTraining = 50; # Numero de veces que se va a entrenar la RNA para cada fold por el hecho de ser no determinístico el entrenamiento

#Si no está generado el dataset pues lo creamos
dataset_name="datasets/iris.data"
if (!isfile(dataset_name))
    (inputs, targets) = getInputs("datasets");
    println("Tamaños en la generación:")
    println(size(inputs))
    println(size(targets))
    write_dataset(dataset_name,inputs,targets)
end

# Cargamos el dataset
dataset = readdlm(dataset_name,',');

# Preparamos las entradas y las salidas deseadas
inputs = convert(Array{Float64,2}, dataset[:,1:6]);             #Array{Float64,2}
targets = oneHotEncoding(convert(Array{Any,1},dataset[:,7]));   #Array{Bool,2}

# Normalizamos las entradas, a pesar de que algunas se vayan a utilizar para test
#normalizeMinMax!(inputs);

numClasses = size(targets,2);
# Nos aseguramos que el numero de clases es mayor que 2, porque en caso contrario no tiene sentido hacer un "one vs all"
#@assert(numClasses>2);

# Normalizamos las entradas, a pesar de que algunas se vayan a utilizar para test
#normalizeMinMax!(inputs);

# Creamos los indices de crossvalidation
crossValidationIndices = crossvalidation(size(inputs,1), numFolds);

# Creamos los vectores para las metricas que se vayan a usar
# En este caso, solo voy a usar precision y F1, en otro problema podrían ser distintas
testAccuracies = Array{Float64,1}(undef, numFolds);
testF1         = Array{Float64,1}(undef, numFolds);

# Para cada fold, entrenamos
for numFold in 1:numFolds

    # Dividimos los datos en entrenamiento y test
    local trainingInputs, testInputs, trainingTargets, testTargets;
    trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
    testInputs        = inputs[crossValidationIndices.==numFold,:];
    trainingTargets   = targets[crossValidationIndices.!=numFold,:];
    testTargets       = targets[crossValidationIndices.==numFold,:];

    # En el caso de entrenar una RNA, este proceso es no determinístico, por lo que es necesario repetirlo para cada fold
    # Para ello, se crean vectores adicionales para almacenar las metricas para cada entrenamiento
    testAccuraciesEachRepetition = Array{Float64,1}(undef, numRepetitionsAANTraining);
    testF1EachRepetition         = Array{Float64,1}(undef, numRepetitionsAANTraining);

    for numTraining in 1:numRepetitionsAANTraining

        # Para el caso de entrenar una RNA con conjunto de validacion, hacemos una división adicional:
        #  dividimos el conjunto de entrenamiento en entrenamiento+validacion
        #  Para ello, hacemos un hold out
        local trainingIndices, validationIndices;
        (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), validationRatio*size(trainingInputs,1)/size(inputs,1));
        # Con estos indices, se pueden crear los vectores finales que vamos a usar para entrenar una RNA

        # Entrenamos la RNA
        local ann;
        ann, = trainClassANN(topology,
        trainingInputs[trainingIndices,:],   trainingTargets[trainingIndices,:],
        trainingInputs[validationIndices,:], trainingTargets[validationIndices,:],
        testInputs,                          testTargets;
        maxEpochs=numMaxEpochs, learningRate=learningRate, maxEpochsVal=maxEpochsVal);


        # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
        (acc, _, _, _, _, _, F1, _) = confusionMatrix(collect(ann(testInputs')'), testTargets);

        # Almacenamos las metricas de este entrenamiento
        testAccuraciesEachRepetition[numTraining] = acc;
        println(string("Acc (numTraining:", numTraining, ") -> ", string(acc)))
        testF1EachRepetition[numTraining]         = F1;

    end;

    # Almacenamos las 2 metricas que usamos en este problema
    testAccuracies[numFold] = mean(testAccuraciesEachRepetition);
    testF1[numFold]         = mean(testF1EachRepetition);

    println("Results in test in fold ", numFold, "/", numFolds, ": accuracy: ", 100*testAccuracies[numFold], " %, F1: ", 100*testF1[numFold], " %");

end;

println("Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ", 100*std(testAccuracies));
println("Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ", 100*std(testF1));
