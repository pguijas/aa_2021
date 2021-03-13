# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 1 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using FileIO;
using DelimitedFiles;

function oneHotEncoding(feature::Array{Any,1})
    classes = unique(feature);
    numClasses = length(classes);
    @assert(numClasses>1)
    if (numClasses==2)
        oneHot = Array{Bool,2}(undef, size(feature,1), 1);
        oneHot[:,1] .= (feature.==classes[1]);
    else
        oneHot = Array{Bool,2}(undef, size(feature,1), numClasses);
        for numClass = 1:numClasses
            oneHot[:,numClass] .= (feature.==classes[numClass]);
        end;
    end;
    return oneHot;
end;
# Sobrecargamos la funcion oneHotEncoding por si acaso pasan un vector de valores booleanos
#  En este caso, el propio vector ya está codificado
# Cuando se llame a la funcion oneHotEncoding, según el tipo del argumento pasado, Julia realizará
#  la llamada a la función de arriba o esta
oneHotEncoding(feature::Array{Bool,1}) = feature;



# -------------------------------------------------------------------------
# Ejemplo de uso de estas funciones:

# Cargamos el dataset
dataset = readdlm("iris.data",',');

# Preparamos las entradas
inputs = dataset[:,1:4];
# Con cualquiera de estas 3 maneras podemos convertir la matriz de entradas de tipo Array{Any,2} en Array{Float64,2}, si los valores son numéricos:
inputs = Float64.(inputs);
inputs = convert(Array{Float64,2},inputs);
inputs = [Float64(x) for x in inputs];
println("Tamaño de la matriz de entradas: ", size(inputs,1), "x", size(inputs,2), " de tipo ", typeof(inputs));

# Preparamos las salidas deseadas codificándolas puesto que son categóricas
targets = dataset[:,5];
println("Longitud del vector de salidas deseadas antes de codificar: ", length(targets), " de tipo ", typeof(targets));
targets = oneHotEncoding(targets);
println("Tamaño de la matriz de salidas deseadas despues de codificar: ", size(targets,1), "x", size(targets,2), " de tipo ", typeof(targets));

# Comprobamos que ambas matrices tienen el mismo número de filas
@assert (size(inputs,1)==size(targets,1)) "Las matrices de entradas y salidas deseadas no tienen el mismo numero de filas"
















# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 2 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using Statistics
using Flux
using Flux.Losses


# -------------------------------------------------------
# Funciones para calcular la precision

accuracy(outputs::Array{Bool,1}, targets::Array{Bool,1}) = mean(outputs.==targets);
function accuracy(outputs::Array{Bool,2}, targets::Array{Bool,2}; dataInRows::Bool=true)
    @assert(all(size(outputs).==size(targets)));
    if (dataInRows)
        # Cada patron esta en cada fila
        if (size(targets,2)==1)
            return accuracy(outputs[:,1], targets[:,1]);
        else
            classComparison = targets .== outputs
            correctClassifications = all(classComparison, dims=2)
            return mean(correctClassifications)
        end;
    else
        # Cada patron esta en cada columna
        if (size(targets,1)==1)
            return accuracy(outputs[1,:], targets[1,:]);
        else
            classComparison = targets .== outputs
            correctClassifications = all(classComparison, dims=1)
            return mean(correctClassifications)
        end;
    end;
end;

accuracy(outputs::Array{Float64,1}, targets::Array{Bool,1}; threshold::Float64=0.5) = accuracy(Array{Bool,1}(outputs.>=threshold), targets);
function accuracy(outputs::Array{Float64,2}, targets::Array{Bool,2}; dataInRows::Bool=true)
    @assert(all(size(outputs).==size(targets)));
    if (dataInRows)
        # Cada patron esta en cada fila
        if (size(targets,2)==1)
            return accuracy(outputs[:,1], targets[:,1]);
        else
            vmax = maximum(outputs, dims=2);
            outputs = Array{Bool,2}(outputs .== vmax);
            return accuracy(outputs, targets);
        end;
    else
        # Cada patron esta en cada columna
        if (size(targets,1)==1)
            return accuracy(outputs[1,:], targets[1,:]);
        else
            vmax = maximum(outputs, dims=1)
            outputs = Array{Bool,2}(outputs .== vmax)
            return accuracy(outputs, targets; dataInRows=false);
        end;
    end;
end;

# Añado estas funciones porque las RR.NN.AA. dan la salida como matrices de valores Float32 en lugar de Float64
# Con estas funciones se pueden usar indistintamente matrices de Float32 o Float64
accuracy(outputs::Array{Float32,1}, targets::Array{Bool,1}; threshold::Float64=0.5) = accuracy(Float64.(outputs), targets; threshold=threshold);
accuracy(outputs::Array{Float32,2}, targets::Array{Bool,2}; dataInRows::Bool=true)  = accuracy(Float64.(outputs), targets; dataInRows=dataInRows);


# -------------------------------------------------------
# Funciones para crear y entrenar una RNA

function buildClassANN(numInputs::Int64, topology::Array{Int64,1}, numOutputs::Int64)
    ann=Chain();
    numInputsLayer = numInputs;
    for numOutputLayers = topology
        ann = Chain(ann..., Dense(numInputsLayer, numOutputLayers, σ));
        numInputsLayer = numOutputLayers;
    end;
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);
    end;
    return ann;
end;

function trainClassANN(topology::Array{Int64,1}, inputs::Array{Float64,2}, targets::Array{Bool,2}; maxEpochs::Int64=1000, minLoss::Float64=0.0, learningRate::Float64=0.1)
    # Se supone que tenemos cada patron en cada fila
    # Comprobamos que el numero de filas (numero de patrones) coincide
    @assert(size(inputs,1)==size(targets,1));
    # Creamos la RNA
    ann = buildClassANN(size(inputs,2), topology, size(targets,2));
    # Definimos la funcion de loss
    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
    # Creamos los vectores con los valores de loss y de precision en cada ciclo
    trainingLosses = Float64[];
    trainingAccuracies = Float64[];
    # Empezamos en el ciclo 0
    numEpoch = 0;

    # Una funcion util para calcular los resultados y mostrarlos por pantalla
    function calculateMetrics()
        # Calculamos el loss. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        trainingLoss = loss(inputs', targets');
        # Calculamos la salida de la RNA. Para ello hay que pasar la matriz de entradas traspuesta (cada patron en una columna). La matriz de salidas tiene un patron en cada columna
        outputs = ann(inputs');
        # Para calcular la precision, ponemos 2 opciones aqui equivalentes:
        #  Pasar las matrices con los datos en las columnas. La matriz de salidas ya tiene un patron en cada columna
        acc = accuracy(outputs, Array{Bool,2}(targets'); dataInRows=false);
        #  Pasar las matrices con los datos en las filas. Hay que trasponer la matriz de salidas de la RNA, puesto que cada dato esta en una fila
        acc = accuracy(Array{Float64,2}(outputs'), targets; dataInRows=true);
        # Mostramos por pantalla el resultado de este ciclo de entrenamiento
        println("Epoch ", numEpoch, ": loss: ", trainingLoss, ", accuracy: ", 100*acc, " %");
        return (trainingLoss, acc)
    end;

    # Calculamos las metricas para el ciclo 0 (sin entrenar nada)
    (trainingLoss, trainingAccuracy) = calculateMetrics();
    #  y almacenamos el valor de loss y precision en este ciclo
    push!(trainingLosses, trainingLoss);
    push!(trainingAccuracies, trainingAccuracy);

    # Entrenamos hasta que se cumpla una condicion de parada
    while (numEpoch<maxEpochs) && (trainingLoss>minLoss)
        # Entrenamos 1 ciclo. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        Flux.train!(loss, params(ann), [(inputs', targets')], ADAM(learningRate));
        # Aumentamos el numero de ciclo en 1
        numEpoch += 1;
        # Calculamos las metricas en este ciclo
        (trainingLoss, trainingAccuracy) = calculateMetrics()
        #  y almacenamos el valor de loss y precision en este ciclo
        push!(trainingLosses, trainingLoss);
        push!(trainingAccuracies, trainingAccuracy);
    end;
    return (ann, trainingLosses, trainingAccuracies);
end;

# -------------------------------------------------------------------------
# Funciones para calcular los parametros de normalizacion y normalizar

# Para calcular los parametros de normalizacion, segun la forma de normalizar que se desee:
calculateMinMaxNormalizationParameters(dataset::Array{Float64,2}; dataInRows=true) =
    ( minimum(dataset, dims=(dataInRows ? 1 : 2)), maximum(dataset, dims=(dataInRows ? 1 : 2)) );

calculateZeroMeanNormalizationParameters(dataset::Array{Float64,2}; dataInRows=true) =
    ( mean(dataset, dims=(dataInRows ? 1 : 2)), std(dataset, dims=(dataInRows ? 1 : 2)) );

# 4 versiones de la funcion para normalizar entre 0 y 1:
#  - Nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - No nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - Nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
#  - No nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
function normalizeMinMax!(dataset::Array{Float64,2}, normalizationParameters::Tuple{Array{Float64,2},Array{Float64,2}}; dataInRows=true)
    min = normalizationParameters[1];
    max = normalizationParameters[2];
    dataset .-= min;
    dataset ./= (max .- min);
    # Si hay algun atributo en el que todos los valores son iguales, se pone a 0
    if (dataInRows)
        dataset[:, vec(min.==max)] .= 0;
    else
        dataset[vec(min.==max), :] .= 0;
    end
end;
normalizeMinMax!(dataset::Array{Float64,2}; dataInRows=true) = normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset; dataInRows=dataInRows); dataInRows=dataInRows);
function normalizeMinMax(dataset::Array{Float64,2}, normalizationParameters::Tuple{Array{Float64,2},Array{Float64,2}}; dataInRows=true)
    newDataset = copy(dataset);
    normalizeMinMax!(newDataset, normalizationParameters; dataInRows=dataInRows);
    return newDataset;
end;
normalizeMinMax(dataset::Array{Float64,2}; dataInRows=true) = normalizeMinMax(dataset, calculateMinMaxNormalizationParameters(dataset; dataInRows=dataInRows); dataInRows=dataInRows);


# 4 versiones similares de la funcion para normalizar de media 0:
#  - Nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - No nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - Nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
#  - No nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
function normalizeZeroMean!(dataset::Array{Float64,2}, normalizationParameters::Tuple{Array{Float64,2},Array{Float64,2}}; dataInRows=true)
    avg  = normalizationParameters[1];
    stnd = normalizationParameters[2];
    dataset .-= avg;
    dataset ./= stnd;
    # Si hay algun atributo en el que todos los valores son iguales, se pone a 0
    if (dataInRows)
        dataset[:, vec(stnd.==0)] .= 0;
    else
        dataset[vec(stnd.==0), :] .= 0;
    end
end;
normalizeZeroMean!(dataset::Array{Float64,2}; dataInRows=true) = normalizeZeroMean!(dataset, calculateZeroMeanNormalizationParameters(dataset; dataInRows=dataInRows); dataInRows=dataInRows);
function normalizeZeroMean(dataset::Array{Float64,2}, normalizationParameters::Tuple{Array{Float64,2},Array{Float64,2}}; dataInRows=true)
    newDataset = copy(dataset);
    normalizeZeroMean!(newDataset, normalizationParameters; dataInRows=dataInRows);
    return newDataset;
end;
normalizeZeroMean(dataset::Array{Float64,2}; dataInRows=true) = normalizeZeroMean(dataset, calculateZeroMeanNormalizationParameters(dataset; dataInRows=dataInRows); dataInRows=dataInRows);




# -------------------------------------------------------------------------
# Ejemplo de uso de estas funciones:

# Parametros principales de la RNA y del proceso de entrenamiento
topology = [4, 3]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento

# Comprobamos que las funciones de normalizar funcionan correctamente
# Normalizacion entre maximo y minimo
newInputs = normalizeMinMax(inputs);
@assert(all(minimum(newInputs, dims=1) .== 0));
@assert(all(maximum(newInputs, dims=1) .== 1));
# Normalizacion de media 0. en este caso, debido a redondeos, la media y desviacion tipica de cada variable no van a dar exactamente 0 y 1 respectivamente. Por eso las comprobaciones se hacen de esta manera
newInputs = normalizeZeroMean(inputs);
@assert(all(abs.(mean(newInputs, dims=1))    .<= 1e-10));
@assert(all(abs.(std( newInputs, dims=1)).-1 .<= 1e-10));

# Finalmente, normalizamos las entradas entre maximo y minimo:
normalizeMinMax!(inputs);
# Y creamos y entrenamos la RNA con los parametros dados
(ann, trainingLosses, trainingAccuracies) = trainClassANN(topology, inputs, targets; maxEpochs=numMaxEpochs, learningRate=learningRate);



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 3 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

function holdOut(N::Int, P::Float64)
    @assert ((P>=0.) & (P<=1.));
    indices = randperm(N);
    numTrainingInstances = Int(round(N*(1-P)));
    return (indices[1:numTrainingInstances], indices[numTrainingInstances+1:end]);
end

function holdOut(N::Int, Pval::Float64, Ptest::Float64)
    @assert ((Pval>=0.) & (Pval<=1.));
    @assert ((Ptest>=0.) & (Ptest<=1.));
    @assert ((Pval+Ptest)<=1.);
    # Primero separamos en entrenamiento+validation y test
    (trainingValidationIndices, testIndices) = holdOut(N, Ptest);
    # Después separamos el conjunto de entrenamiento+validation
    (trainingIndices, validationIndices) = holdOut(length(trainingValidationIndices), Pval*N/length(trainingValidationIndices))
    return (trainingValidationIndices[trainingIndices], trainingValidationIndices[validationIndices], testIndices);
end;


# Funcion para entrenar RR.NN.AA. con conjuntos de entrenamiento y test
# Es la funcion anterior, modificada para calcular errores en el conjunto de test
function trainClassANN(topology::Array{Int64,1},
    trainingInputs::Array{Float64,2}, trainingTargets::Array{Bool,2},
    testInputs::Array{Float64,2},     testTargets::Array{Bool,2};
    maxEpochs::Int64=1000, minLoss::Float64=0.0, learningRate::Float64=0.1, showText::Bool=false)

    # Se supone que tenemos cada patron en cada fila
    # Comprobamos que el numero de filas (numero de patrones) coincide tanto en entrenamiento como en test
    @assert(size(trainingInputs,1)==size(trainingTargets,1));
    @assert(size(testInputs,1)==size(testTargets,1));
    # Comprobamos que el numero de columnas coincide en los grupos de entrenamiento y test
    @assert(size(trainingInputs,2)==size(testInputs,2));
    @assert(size(trainingTargets,2)==size(testTargets,2));
    # Creamos la RNA
    ann = buildClassANN(size(trainingInputs,2), topology, size(trainingTargets,2));
    # Definimos la funcion de loss
    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
    # Creamos los vectores con los valores de loss y de precision en cada ciclo
    trainingLosses = Float64[];
    trainingAccuracies = Float64[];
    testLosses = Float64[];
    testAccuracies = Float64[];
    # Empezamos en el ciclo 0
    numEpoch = 0;

    # Una funcion util para calcular los resultados y mostrarlos por pantalla
    function calculateMetrics()
        # Calculamos el loss en entrenamiento y test. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        trainingLoss = loss(trainingInputs', trainingTargets');
        testLoss     = loss(testInputs',     testTargets');
        # Calculamos la salida de la RNA en entrenamiento y test. Para ello hay que pasar la matriz de entradas traspuesta (cada patron en una columna). La matriz de salidas tiene un patron en cada columna
        trainingOutputs = ann(trainingInputs');
        testOutputs     = ann(testInputs');
        # Para calcular la precision, ponemos 2 opciones aqui equivalentes:
        #  Pasar las matrices con los datos en las columnas. La matriz de salidas ya tiene un patron en cada columna
        trainingAcc = accuracy(trainingOutputs, Array{Bool,2}(trainingTargets'); dataInRows=false);
        testAcc     = accuracy(testOutputs,     Array{Bool,2}(testTargets');     dataInRows=false);
        #  Pasar las matrices con los datos en las filas. Hay que trasponer la matriz de salidas de la RNA, puesto que cada dato esta en una fila
        trainingAcc = accuracy(Array{Float64,2}(trainingOutputs'), trainingTargets; dataInRows=true);
        testAcc     = accuracy(Array{Float64,2}(testOutputs'),     testTargets;     dataInRows=true);
        # Mostramos por pantalla el resultado de este ciclo de entrenamiento si nos lo han indicado
        if showText
            println("Epoch ", numEpoch, ": Training loss: ", trainingLoss, ", accuracy: ", 100*trainingAcc, " % - Test loss: ", testLoss, ", accuracy: ", 100*testAcc, " %");
        end;
        return (trainingLoss, trainingAcc, testLoss, testAcc)
    end;

    # Calculamos las metricas para el ciclo 0 (sin entrenar nada)
    (trainingLoss, trainingAccuracy, testLoss, testAccuracy) = calculateMetrics();
    #  y almacenamos los valores de loss y precision en este ciclo
    push!(trainingLosses,      trainingLoss);
    push!(testLosses,          testLoss);
    push!(trainingAccuracies,  trainingAccuracy);
    push!(testAccuracies,      testAccuracy);

    # Entrenamos hasta que se cumpla una condicion de parada
    while (numEpoch<maxEpochs) && (trainingLoss>minLoss)
        # Entrenamos 1 ciclo. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        Flux.train!(loss, params(ann), [(trainingInputs', trainingTargets')], ADAM(learningRate));
        # Aumentamos el numero de ciclo en 1
        numEpoch += 1;
        # Calculamos las metricas en este ciclo
        (trainingLoss, trainingAccuracy, testLoss, testAccuracy) = calculateMetrics();
        #  y almacenamos los valores de loss y precision en este ciclo
        push!(trainingLosses,     trainingLoss);
        push!(trainingAccuracies, trainingAccuracy);
        push!(testLosses,         testLoss);
        push!(testAccuracies,     testAccuracy);
    end;
    return (ann, trainingLosses, testLosses, trainingAccuracies, testAccuracies);
end;



# Funcion para entrenar RR.NN.AA. con conjuntos de entrenamiento, validacion y test
# Es la funcion anterior, modificada para calcular errores en el conjunto de validacion, y parar el entrenamiento si es necesario
function trainClassANN(topology::Array{Int64,1},
    trainingInputs::Array{Float64,2},   trainingTargets::Array{Bool,2},
    validationInputs::Array{Float64,2}, validationTargets::Array{Bool,2},
    testInputs::Array{Float64,2},       testTargets::Array{Bool,2};
    maxEpochs::Int64=1000, minLoss::Float64=0.0, learningRate::Float64=0.1, maxEpochsVal::Int64=6, showText::Bool=false)

    # Se supone que tenemos cada patron en cada fila
    # Comprobamos que el numero de filas (numero de patrones) coincide tanto en entrenamiento como en validation como test
    @assert(size(trainingInputs,1)==size(trainingTargets,1));
    @assert(size(validationInputs,1)==size(validationTargets,1));
    @assert(size(testInputs,1)==size(testTargets,1));
    # Comprobamos que el numero de columnas coincide en los grupos de entrenamiento, validacion y test
    @assert(size(trainingInputs,2)==size(validationInputs,2)==size(testInputs,2));
    @assert(size(trainingTargets,2)==size(validationTargets,2)==size(testTargets,2));
    # Creamos la RNA
    ann = buildClassANN(size(trainingInputs,2), topology, size(trainingTargets,2));
    # Definimos la funcion de loss
    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
    # Creamos los vectores con los valores de loss y de precision en cada ciclo
    trainingLosses = Float64[];
    trainingAccuracies = Float64[];
    validationLosses = Float64[];
    validationAccuracies = Float64[];
    testLosses = Float64[];
    testAccuracies = Float64[];

    # Empezamos en el ciclo 0
    numEpoch = 0;

    # Una funcion util para calcular los resultados y mostrarlos por pantalla
    function calculateMetrics()
        # Calculamos el loss en entrenamiento y test. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        trainingLoss   = loss(trainingInputs',   trainingTargets');
        validationLoss = loss(validationInputs', validationTargets');
        testLoss       = loss(testInputs',       testTargets');
        # Calculamos la salida de la RNA en entrenamiento y test. Para ello hay que pasar la matriz de entradas traspuesta (cada patron en una columna). La matriz de salidas tiene un patron en cada columna
        trainingOutputs   = ann(trainingInputs');
        validationOutputs = ann(validationInputs');
        testOutputs       = ann(testInputs');
        # Para calcular la precision, ponemos 2 opciones aqui equivalentes:
        #  Pasar las matrices con los datos en las columnas. La matriz de salidas ya tiene un patron en cada columna
        trainingAcc   = accuracy(trainingOutputs,   Array{Bool,2}(trainingTargets');   dataInRows=false);
        validationAcc = accuracy(validationOutputs, Array{Bool,2}(validationTargets'); dataInRows=false);
        testAcc       = accuracy(testOutputs,       Array{Bool,2}(testTargets');       dataInRows=false);
        #  Pasar las matrices con los datos en las filas. Hay que trasponer la matriz de salidas de la RNA, puesto que cada dato esta en una fila
        trainingAcc   = accuracy(Array{Float64,2}(trainingOutputs'),   trainingTargets;   dataInRows=true);
        validationAcc = accuracy(Array{Float64,2}(validationOutputs'), validationTargets; dataInRows=true);
        testAcc       = accuracy(Array{Float64,2}(testOutputs'),       testTargets;       dataInRows=true);
        # Mostramos por pantalla el resultado de este ciclo de entrenamiento si nos lo han indicado
        if showText
            println("Epoch ", numEpoch, ": Training loss: ", trainingLoss, ", accuracy: ", 100*trainingAcc, " % - Validation loss: ", validationLoss, ", accuracy: ", 100*validationAcc, " % - Test loss: ", testLoss, ", accuracy: ", 100*testAcc, " %");
        end;
        return (trainingLoss, trainingAcc, validationLoss, validationAcc, testLoss, testAcc)
    end;

    # Calculamos las metricas para el ciclo 0 (sin entrenar nada)
    (trainingLoss, trainingAccuracy, validationLoss, validationAccuracy, testLoss, testAccuracy) = calculateMetrics();
    #  y almacenamos los valores de loss y precision en este ciclo
    push!(trainingLosses,       trainingLoss);
    push!(trainingAccuracies,   trainingAccuracy);
    push!(validationLosses,     validationLoss);
    push!(validationAccuracies, validationAccuracy);
    push!(testLosses,           testLoss);
    push!(testAccuracies,       testAccuracy);

    # Numero de ciclos sin mejorar el error de validacion y el mejor error de validation encontrado hasta el momento
    numEpochsValidation = 0; bestValidationLoss = validationLoss;
    # Cual es la mejor ann que se ha conseguido
    bestANN = deepcopy(ann);

    # Entrenamos hasta que se cumpla una condicion de parada
    while (numEpoch<maxEpochs) && (trainingLoss>minLoss) && (numEpochsValidation<maxEpochsVal)
        # Entrenamos 1 ciclo. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        Flux.train!(loss, params(ann), [(trainingInputs', trainingTargets')], ADAM(learningRate));
        # Aumentamos el numero de ciclo en 1
        numEpoch += 1;
        # Calculamos las metricas en este ciclo
        (trainingLoss, trainingAccuracy, validationLoss, validationAccuracy, testLoss, testAccuracy) = calculateMetrics();
        #  y almacenamos los valores de loss y precision en este ciclo
        push!(trainingLosses,       trainingLoss);
        push!(trainingAccuracies,   trainingAccuracy);
        push!(validationLosses,     validationLoss);
        push!(validationAccuracies, validationAccuracy);
        push!(testLosses,           testLoss);
        push!(testAccuracies,       testAccuracy);
        # Aplicamos la parada temprana
        if (validationLoss<bestValidationLoss)
            bestValidationLoss = validationLoss;
            numEpochsValidation = 0;
            bestANN = deepcopy(ann);
        else
            numEpochsValidation += 1;
        end;
    end;
    return (bestANN, trainingLosses, validationLosses, testLosses, trainingAccuracies, validationAccuracies, testAccuracies);
end;




# -------------------------------------------------------------------------
# Ejemplo de uso de estas funciones, con conjuntos de entrenamiento y test:

# Parametros principales de la RNA y del proceso de entrenamiento
topology = [4, 3]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
testRatio = 0.2; # Porcentaje de patrones que se usaran para test

# Cargamos el dataset
dataset = readdlm("iris.data",',');
# Preparamos las entradas y las salidas deseadas
inputs = convert(Array{Float64,2}, dataset[:,1:4]);
targets = oneHotEncoding(dataset[:,5]);

# Creamos los indices de entrenamiento y test
(trainingIndices, testIndices) = holdOut(size(inputs,1), testRatio);

# Dividimos los datos
trainingInputs  = inputs[trainingIndices,:];
testInputs      = inputs[testIndices,:];
trainingTargets = targets[trainingIndices,:];
testTargets     = targets[testIndices,:];

# Calculamos los valores de normalizacion solo del conjunto de entrenamiento
normalizationParams = calculateMinMaxNormalizationParameters(trainingInputs);

# Normalizamos las entradas entre maximo y minimo de forma separada para entrenamiento y test, con los parametros hallados anteriormente
normalizeMinMax!(trainingInputs, normalizationParams);
normalizeMinMax!(testInputs,     normalizationParams);

# Y creamos y entrenamos la RNA con los parametros dados
(ann, trainingLosses, trainingAccuracies) = trainClassANN(topology,
    trainingInputs, trainingTargets,
    testInputs,     testTargets;
    maxEpochs=numMaxEpochs, learningRate=learningRate, showText=true);



# -------------------------------------------------------------------------
# Ejemplo de uso de estas funciones, con conjuntos de entrenamiento, validacion y test:

# Parametros principales de la RNA y del proceso de entrenamiento
topology = [4, 3]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
validationRatio = 0.2; # Porcentaje de patrones que se usaran para validacion
testRatio = 0.2; # Porcentaje de patrones que se usaran para test
maxEpochsVal = 6; # Numero de ciclos en los que si no se mejora el loss en el conjunto de validacion, se para el entrenamiento

# Cargamos el dataset
dataset = readdlm("iris.data",',');
# Preparamos las entradas y las salidas deseadas
inputs = convert(Array{Float64,2}, dataset[:,1:4]);
targets = oneHotEncoding(dataset[:,5]);

# Normalizamos las entradas, a pesar de que algunas se vayan a utilizar para test
normalizeMinMax!(inputs);

# Creamos los indices de entrenamiento, validacion y test
(trainingIndices, validationIndices, testIndices) = holdOut(size(inputs,1), validationRatio, testRatio);

# Dividimos los datos
trainingInputs    = inputs[trainingIndices,:];
validationInputs  = inputs[validationIndices,:];
testInputs        = inputs[testIndices,:];
trainingTargets   = targets[trainingIndices,:];
validationTargets = targets[validationIndices,:];
testTargets       = targets[testIndices,:];

# Y creamos y entrenamos la RNA con los parametros dados
(ann, trainingLosses, trainingAccuracies) = trainClassANN(topology,
    trainingInputs,   trainingTargets,
    validationInputs, validationTargets,
    testInputs,       testTargets;
    maxEpochs=numMaxEpochs, learningRate=learningRate, maxEpochsVal=maxEpochsVal, showText=true);
