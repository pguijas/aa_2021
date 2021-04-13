include("modulos/bondad.jl")
include("modulos/graphics.jl")
include("modulos/datasets.jl")
include("modulos/svm.jl")
include("modulos/attributes_from_dataset.jl")

using Flux;

# Parametros del SVM
kernel = "rbf";
kernelDegree = 3;
kernelGamma = 2;
C=1;

#Si no está generado el dataset pues lo creamos
dataset_name="datasets/faces.data"
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

# Creamos los indices de entrenamiento, validacion y test
(trainingIndices, validationIndices, testIndices) = holdOut(size(inputs,1), validationRatio, testRatio);

# Dividimos los datos
trainingInputs    = inputs[trainingIndices,:];
validationInputs  = inputs[validationIndices,:];
testInputs        = inputs[testIndices,:];
trainingTargets   = targets[trainingIndices,:];
validationTargets = targets[validationIndices,:];
testTargets       = targets[testIndices,:];

println("Empezando el entrenamiento.")
println()

# Entrenamento SVM
svm = trainClassSVM(trainingInputs, trainingTargets, testInputs, testTargets;
            kernel=kernel, kernelDegree=kernelDegree, kernelGamma=kernelGamma, C=C);

# esto podríamos meterlo dentro de la matriz de confusión
testOutputs=copy(convert(Array{Float64,2},predict(svm,inputs)')');
testOutputs=classifyOutputs(testOutputs); #Array{Bool,2}
printConfusionMatrix(testOutputs, targets);
