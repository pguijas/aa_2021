include("modulos/bondad.jl")
include("modulos/graphics.jl")
include("modulos/datasets.jl")
include("modulos/rna.jl")
include("modulos/attributes_from_dataset.jl")

using CUDA;
using Flux: gpu;

#RECORDAR:
#   Para revisar que esté todo bien mostrar gráficos del codigo del
#   profe y el nuestro, conjunto test tintinea un poco

#===============================================================================
DUDAS
    - Héctor: ¿Necesitamos medir la varianza? La varianza solo es la desviación
        típica elevada al cuadrado, creo que estamos midiendo lo mismo. La
        media mide la cantidad de color que tiene una imagen, y la desviacion
        típica la dispersión del color, la varianza sigue midiendo la dispersión
        del color, solo que al cuadrado, le estamos dando más importancia a la
        desviación típica que a la media y la red va a aprender que es + importante
        la dispersión que la media, cuando deberían de tener la misma importancia.
===============================================================================#

# Parametros principales de la RNA y del proceso de entrenamiento
topology = [4, 3]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
validationRatio = 0.2; # Porcentaje de patrones que se usaran para validacion
testRatio = 0.2; # Porcentaje de patrones que se usaran para test
maxEpochsVal = 15; # Numero de ciclos en los que si no se mejora el loss en el conjunto de validacion, se para el entrenamiento

#Si no está generado el dataset pues lo creamos
dataset_name="datasets/faces.data"
if (!isfile(dataset_name))
    (inputs, targets) = getInputs("datasets");
    println("Tamaños en la generación:")
    println(size(inputs))
    println(size(targets))
    write_dataset(dataset_name,inputs,targets)
end

# Héctor: A mi me da bastantes errores la función de escribir el dataset,
#   quizás podríamos mejorarla.

# Cargamos el dataset
dataset = readdlm(dataset_name,',');

# Preparamos las entradas y las salidas deseadas
inputs = convert(Array{Float64,2}, dataset[:,1:6]);
targets = oneHotEncoding(convert(Array{Any,1},dataset[:,7]));

println()
print("Inputs: ")
print(size(inputs))
print(", ")
print(typeof(inputs))

println()
print("Targets: ")
print(size(targets))
print(", ")
println(typeof(targets))

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
# Y creamos y entrenamos la RNA con los parametros dados
(ann, trainingLosses, validationLosses, testLosses, trainingAccuracies,
validationAccuracies, testAccuracies) = trainClassANN(topology, trainingInputs, trainingTargets, validationInputs,
                                            validationTargets, testInputs, testTargets; maxEpochs=numMaxEpochs,
                                            learningRate=learningRate, maxEpochsVal=maxEpochsVal, showText=true) |> gpu;

print_train_results(trainingLosses, validationLosses, testLosses, trainingAccuracies, validationAccuracies, testAccuracies);

#Resultados finales sobre todos los patrones:
println(accuracy(Array{Float64,2}(ann(inputs')'),targets));

# esto podríamos meterlo dentro de la matriz de confusión
outputs=((ann(inputs')').>0.5);
outputs=convert(Array{Bool,2},outputs);


confusionMatrix(outputs,targets,true);

#println("Resultados en el conjunto de entrenamiento:")
#trainingOutputs = (collect(ann(trainingInputs')')).>0.5;
#printConfusionMatrix2(convert(Array{Bool,2},trainingOutputs), trainingTargets; weighted=true);
#println("Resultados en el conjunto de validación:")
#validationOutputs = (collect(ann(validationInputs')')).>0.5;
#printConfusionMatrix2(convert(Array{Bool,2},validationOutputs), validationTargets; weighted=true);
#println("Resultados en el conjunto de test:")
#testOutputs = (collect(ann(testInputs')')).>0.5;
#printConfusionMatrix2(convert(Array{Bool,2},testOutputs), testTargets; weighted=true);
#println("Resultados globales:")
#outputs = (collect(ann(inputs')')).>0.5;
#printConfusionMatrix2(convert(Array{Bool,2},outputs), targets; weighted=true);
