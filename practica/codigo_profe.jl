# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 1 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using FileIO;
using DelimitedFiles;

# Funcion para realizar la codificacion, recibe el vector de caracteristicas (uno por patron), y las clases
function oneHotEncoding(feature::Array{Any,1}, classes::Array{Any,1})
    # Primero se comprueba que todos los elementos del vector esten en el vector de clases (linea adaptada del final de la practica 4)
    @assert(all([in(value, classes) for value in feature]));
    numClasses = length(classes);
    @assert(numClasses>1)
    if (numClasses==2)
        # Si solo hay dos clases, se devuelve una matriz con una columna
        oneHot = Array{Bool,2}(undef, size(feature,1), 1);
        oneHot[:,1] .= (feature.==classes[1]);
    else
        # Si hay mas de dos clases se devuelve una matriz con una columna por clase
        oneHot = Array{Bool,2}(undef, size(feature,1), numClasses);
        for numClass = 1:numClasses
            oneHot[:,numClass] .= (feature.==classes[numClass]);
        end;
    end;
    return oneHot;
end;
# Esta funcion es similar a la anterior, pero si no es especifican las clases, se toman de la propia variable
oneHotEncoding(feature::Array{Any,1}) = oneHotEncoding(feature::Array{Any,1}, unique(feature));
# Sobrecargamos la funcion oneHotEncoding por si acaso pasan un vector de valores booleanos
#  En este caso, el propio vector ya está codificado
oneHotEncoding(feature::Array{Bool,1}) = feature;
# Cuando se llame a la funcion oneHotEncoding, según el tipo del argumento pasado, Julia realizará
#  la llamada a la función correspondiente



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
# Funciones auxiliar que permite transformar una matriz de
#  valores reales con las salidas del clasificador o clasificadores
#  en una matriz de valores booleanos con la clase en la que sera clasificada
function classifyOutputs(outputs::Array{Float64,2}; dataInRows::Bool=true)
    # Miramos donde esta el valor mayor de cada instancia con la funcion findmax
    (_,indicesMaxEachInstance) = findmax(outputs, dims= dataInRows ? 2 : 1);
    # Creamos la matriz de valores booleanos con valores inicialmente a false y asignamos esos indices a true
    outputsBoolean = Array{Bool,2}(falses(size(outputs)));
    outputsBoolean[indicesMaxEachInstance] .= true;
    # Comprobamos que efectivamente cada patron solo este clasificado en una clase
    @assert(all(sum(outputsBoolean, dims=dataInRows ? 2 : 1).==1));
    return outputsBoolean;
end;



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
            return accuracy(classifyOutputs(outputs; dataInRows=true), targets);
        end;
    else
        # Cada patron esta en cada columna
        if (size(targets,1)==1)
            return accuracy(outputs[1,:], targets[1,:]);
        else
            return accuracy(classifyOutputs(outputs; dataInRows=false), targets);
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

# Cargamos el dataset
dataset = readdlm("iris.data",',');
# Preparamos las entradas y las salidas deseadas
inputs = convert(Array{Float64,2}, dataset[:,1:4]);
targets = oneHotEncoding(dataset[:,5]);

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












# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 4 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

function confusionMatrix(outputs::Array{Bool,1}, targets::Array{Bool,1})
    @assert(length(outputs)==length(targets));
    @assert(length(outputs)==length(targets));
    # Para calcular la precision y la tasa de error, se puede llamar a las funciones definidas en la practica 2
    acc         = accuracy(outputs, targets); # Precision, definida previamente en una practica anterior
    errorRate   = 1 - acc;
    recall      = mean(  outputs[  targets]); # Sensibilidad
    specificity = mean(.!outputs[.!targets]); # Especificidad
    precision   = mean(  targets[  outputs]); # Valor predictivo positivo
    NPV         = mean(.!targets[.!outputs]); # Valor predictivo negativo
    # Controlamos que algunos casos pueden ser NaN, y otros no
    @assert(!isnan(recall) && !isnan(specificity));
    precision   = isnan(precision) ? 0 : precision;
    NPV         = isnan(NPV) ? 0 : NPV;
    # Calculamos F1
    F1          = (recall==precision==0.) ? 0. : 2*(recall*precision)/(recall+precision);
    # Reservamos memoria para la matriz de confusion
    confMatrix = Array{Int64,2}(undef, 2, 2);
    # Ponemos en las filas los que pertenecen a cada clase (targets) y en las columnas los clasificados (outputs)
    #  Primera fila/columna: negativos
    #  Segunda fila/columna: positivos
    # Primera fila: patrones de clase negativo, clasificados como negativos o positivos
    confMatrix[1,1] = sum(.!targets .& .!outputs); # VN
    confMatrix[1,2] = sum(.!targets .&   outputs); # FP
    # Segunda fila: patrones de clase positiva, clasificados como negativos o positivos
    confMatrix[2,1] = sum(  targets .& .!outputs); # FN
    confMatrix[2,2] = sum(  targets .&   outputs); # VP
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
end;

confusionMatrix(outputs::Array{Float64,1}, targets::Array{Bool,1}; threshold::Float64=0.5) = confusionMatrix(Array{Bool,1}(outputs.>=threshold), targets);


function confusionMatrix(outputs::Array{Bool,2}, targets::Array{Bool,2}; weighted::Bool=true)
    @assert(size(outputs)==size(targets));
    numClasses = size(targets,2);
    # Nos aseguramos de que no hay dos columnas
    @assert(numClasses!=2);
    if (numClasses==1)
        return confusionMatrix(outputs[:,1], targets[:,1]);
    else
        # Nos aseguramos de que en cada fila haya uno y sólo un valor a true
        @assert(all(sum(outputs, dims=2).==1));
        # Reservamos memoria para las metricas de cada clase, inicializandolas a 0 porque algunas posiblemente no se calculen
        recall      = zeros(numClasses);
        specificity = zeros(numClasses);
        precision   = zeros(numClasses);
        NPV         = zeros(numClasses);
        F1          = zeros(numClasses);
        # Reservamos memoria para la matriz de confusion
        confMatrix  = Array{Int64,2}(undef, numClasses, numClasses);
        # Calculamos el numero de patrones de cada clase
        numInstancesFromEachClass = vec(sum(targets, dims=1));
        # Calculamos las metricas para cada clase, esto se haria con un bucle similar a "for numClass in 1:numClasses" que itere por todas las clases
        #  Sin embargo, solo hacemos este calculo para las clases que tengan algun patron
        #  Puede ocurrir que alguna clase no tenga patrones como consecuencia de haber dividido de forma aleatoria el conjunto de patrones entrenamiento/test
        #  En aquellas clases en las que no haya patrones, los valores de las metricas seran 0 (los vectores ya estan asignados), y no se tendran en cuenta a la hora de unir estas metricas
        for numClass in findall(numInstancesFromEachClass.>0)
            # Calculamos las metricas de cada problema binario correspondiente a cada clase y las almacenamos en los vectores correspondientes
            (_, _, recall[numClass], specificity[numClass], precision[numClass], NPV[numClass], F1[numClass], _) = confusionMatrix(outputs[:,numClass], targets[:,numClass]);
        end;

        # Reservamos memoria para la matriz de confusion
        confMatrix = Array{Int64,2}(undef, numClasses, numClasses);
        # Calculamos la matriz de confusión haciendo un bucle doble que itere sobre las clases
        for numClassTarget in 1:numClasses, numClassOutput in 1:numClasses
            # Igual que antes, ponemos en las filas los que pertenecen a cada clase (targets) y en las columnas los clasificados (outputs)
            confMatrix[numClassTarget, numClassOutput] = sum(targets[:,numClassTarget] .& outputs[:,numClassOutput]);
        end;

        # Aplicamos las forma de combinar las metricas macro o weighted
        if weighted
            # Calculamos los valores de ponderacion para hacer el promedio
            weights = numInstancesFromEachClass./sum(numInstancesFromEachClass);
            recall      = sum(weights.*recall);
            specificity = sum(weights.*specificity);
            precision   = sum(weights.*precision);
            NPV         = sum(weights.*NPV);
            F1          = sum(weights.*F1);
        else
            # No realizo la media tal cual con la funcion mean, porque puede haber clases sin instancias
            #  En su lugar, realizo la media solamente de las clases que tengan instancias
            numClassesWithInstances = sum(numInstancesFromEachClass.>0);
            recall      = sum(recall)/numClassesWithInstances;
            specificity = sum(specificity)/numClassesWithInstances;
            precision   = sum(precision)/numClassesWithInstances;
            NPV         = sum(NPV)/numClassesWithInstances;
            F1          = sum(F1)/numClassesWithInstances;
        end;
        # Precision y tasa de error las calculamos con las funciones definidas previamente
        acc = accuracy(outputs, targets; dataInRows=true);
        errorRate = 1 - acc;

        return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
    end;
end;


function confusionMatrix(outputs::Array{Any,1}, targets::Array{Any,1}; weighted::Bool=true)
    # Comprobamos que todas las clases de salida esten dentro de las clases de las salidas deseadas
    @assert(all([in(output, unique(targets)) for output in outputs]));
    classes = unique(targets);
    # Es importante calcular el vector de clases primero y pasarlo como argumento a las 2 llamadas a oneHotEncoding para que el orden de las clases sea el mismo en ambas matrices
    return confusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted);
end;

confusionMatrix(outputs::Array{Float64,2}, targets::Array{Bool,2}; weighted::Bool=true) = confusionMatrix(classifyOutputs(outputs), targets; weighted=weighted);

# De forma similar a la anterior, añado estas funcion porque las RR.NN.AA. dan la salida como matrices de valores Float32 en lugar de Float64
# Con estas funcion se pueden usar indistintamente matrices de Float32 o Float64
confusionMatrix(outputs::Array{Float32,2}, targets::Array{Bool,2}; weighted::Bool=true) = confusionMatrix(convert(Array{Float64,2}, outputs), targets; weighted=weighted);
printConfusionMatrix(outputs::Array{Float32,2}, targets::Array{Bool,2}; weighted::Bool=true) = printConfusionMatrix(convert(Array{Float64,2}, outputs), targets; weighted=weighted);

# Funciones auxiliares para visualizar por pantalla la matriz de confusion y las metricas que se derivan de ella
function printConfusionMatrix(outputs::Array{Bool,2}, targets::Array{Bool,2}; weighted::Bool=true)
    (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets; weighted=weighted);
    numClasses = size(confMatrix,1);
    writeHorizontalLine() = (for i in 1:numClasses+1 print("--------") end; println(""); );
    writeHorizontalLine();
    print("\t| ");
    if (numClasses==2)
        println(" - \t + \t|");
    else
        print.("Cl. ", 1:numClasses, "\t| ");
    end;
    println("");
    writeHorizontalLine();
    for numClassTarget in 1:numClasses
        # print.(confMatrix[numClassTarget,:], "\t");
        if (numClasses==2)
            print(numClassTarget == 1 ? " - \t| " : " + \t| ");
        else
            print("Cl. ", numClassTarget, "\t| ");
        end;
        print.(confMatrix[numClassTarget,:], "\t| ");
        println("");
        writeHorizontalLine();
    end;
    println("Accuracy: ", acc);
    println("Error rate: ", errorRate);
    println("Recall: ", recall);
    println("Specificity: ", specificity);
    println("Precision: ", precision);
    println("Negative predictive value: ", NPV);
    println("F1-score: ", F1);
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
end;
printConfusionMatrix(outputs::Array{Float64,2}, targets::Array{Bool,2}; weighted::Bool=true) =  printConfusionMatrix(classifyOutputs(outputs), targets; weighted=weighted)


# -------------------------------------------------------------------------
# Para probar estas funciones, partimos de los resultados del entrenamiento de la practica anterior

println("Results in the training set:")
trainingOutputs = collect(ann(trainingInputs')');
printConfusionMatrix(trainingOutputs, trainingTargets; weighted=true);
println("Results in the validation set:")
validationOutputs = collect(ann(validationInputs')');
printConfusionMatrix(validationOutputs, validationTargets; weighted=true);
println("Results in the test set:")
testOutputs = collect(ann(testInputs')');
printConfusionMatrix(testOutputs, testTargets; weighted=true);
println("Results in the whole dataset:")
outputs = collect(ann(inputs')');
printConfusionMatrix(outputs, targets; weighted=true);



# -------------------------------------------------------------------------
# Estrategia "uno contra todos" y código de ejemplo:

function oneVSall(inputs::Array{Float64,2}, targets::Array{Bool,2})
    numClasses = size(targets,2);
    # Nos aseguramos de que hay mas de dos clases
    @assert(numClasses>2);
    outputs = Array{Float64,2}(undef, size(inputs,1), numClasses);
    for numClass in 1:numClasses
        model = fit(inputs, targets[:,[numClass]]);
        outputs[:,numClass] .= model(inputs);
    end;
    # Aplicamos la funcion softmax
    outputs = collect(softmax(outputs')');
    # Convertimos a matriz de valores booleanos
    outputs = classifyOutputs(outputs);
    classComparison = (targets .== outputs);
    correctClassifications = all(classComparison, dims=2);
    return mean(correctClassifications);
end;



# A continuacion se muestra de forma practica como se podria usar este esquema de one vs all entrenando RRNNAA en este problema muticlase (flores iris)
# IMPORTANTE: con RR.NN.AA. no es necesario utilizar una estrategia "one vs all" porque ya realiza multiclase

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

numClasses = size(targets,2);
# Nos aseguramos que el numero de clases es mayor que 2, porque en caso contrario no tiene sentido hacer un "one vs all"
@assert(numClasses>2);

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

# Reservamos memoria para las matrices de salidas de entrenamiento, validacion y test
# En lugar de hacer 3 matrices, voy a hacerlo en una sola con todos los datos
outputs = Array{Float64,2}(undef, size(inputs,1), numClasses);

# Y creamos y entrenamos la RNA con los parametros dados para cada una de las clases
for numClass = 1:numClasses

    # A partir de ahora, no vamos a mostrar por pantalla el resultado de cada ciclo del entrenamiento de la RNA (no vamos a poner el showText=true)
    local ann;
    ann, = trainClassANN(topology,
        trainingInputs,   trainingTargets[:,[numClass]],
        validationInputs, validationTargets[:,[numClass]],
        testInputs,       testTargets[:,[numClass]];
        maxEpochs=numMaxEpochs, learningRate=learningRate, maxEpochsVal=maxEpochsVal);

    # Aplicamos la RNA para calcular las salidas para esta clase concreta y las guardamos en la columna correspondiente de la matriz
    outputs[:,numClass] = ann(inputs')';

end;

# A estas 3 matrices de resultados le pasamos la funcion softmax
# Esto es opcional, y nos vale para poder interpretar la salida de cada modelo como la probabilidad de pertenencia de un patron a una clase concreta
outputs = collect(softmax(outputs')');

# Mostramos las matrices de confusion y las metricas
println("Results in the training set:")
printConfusionMatrix(outputs[trainingIndices,:], trainingTargets; weighted=true);
println("Results in the validation set:")
printConfusionMatrix(outputs[validationIndices,:], validationTargets; weighted=true);
println("Results in the test set:")
printConfusionMatrix(outputs[testIndices,:], testTargets; weighted=true);
println("Results in the whole dataset:")
printConfusionMatrix(outputs, targets; weighted=true);









# ----------------------------------------------------------------------------------------------
# ------------------------------------- Practica 5 ---------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    indices = repeat(1:k, Int64(ceil(N/k)));
    indices = indices[1:N];
    shuffle!(indices);
    return indices;
end;


# -------------------------------------------------------------------------
# Código de prueba:

# Fijamos la semilla aleatoria para poder repetir los experimentos
seed!(1);

# Parametros principales de la RNA y del proceso de entrenamiento
topology = [4, 3]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
numFolds = 10;
validationRatio = 0; # Porcentaje de patrones que se usaran para validacion. Puede ser 0, para no usar validacion
maxEpochsVal = 6; # Numero de ciclos en los que si no se mejora el loss en el conjunto de validacion, se para el entrenamiento
numRepetitionsAANTraining = 50; # Numero de veces que se va a entrenar la RNA para cada fold por el hecho de ser no determinístico el entrenamiento

# Cargamos el dataset
dataset = readdlm("iris.data",',');
# Preparamos las entradas y las salidas deseadas
inputs = convert(Array{Float64,2}, dataset[:,1:4]);
targets = oneHotEncoding(dataset[:,5]);

numClasses = size(targets,2);
# Nos aseguramos que el numero de clases es mayor que 2, porque en caso contrario no tiene sentido hacer un "one vs all"
@assert(numClasses>2);

# Normalizamos las entradas, a pesar de que algunas se vayan a utilizar para test
normalizeMinMax!(inputs);

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

        if validationRatio>0

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

        else

            # Si no se desea usar conjunto de validacion, se entrena unicamente con conjuntos de entrenamiento y test
            local ann;
            ann, = trainClassANN(topology,
                trainingInputs, trainingTargets,
                testInputs,     testTargets;
                maxEpochs=numMaxEpochs, learningRate=learningRate);

        end;

        # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
        (acc, _, _, _, _, _, F1, _) = confusionMatrix(collect(ann(testInputs')'), testTargets);

        # Almacenamos las metricas de este entrenamiento
        testAccuraciesEachRepetition[numTraining] = acc;
        testF1EachRepetition[numTraining]         = F1;

    end;

    # Almacenamos las 2 metricas que usamos en este problema
    testAccuracies[numFold] = mean(testAccuraciesEachRepetition);
    testF1[numFold]         = mean(testF1EachRepetition);

    println("Results in test in fold ", numFold, "/", numFolds, ": accuracy: ", 100*testAccuracies[numFold], " %, F1: ", 100*testF1[numFold], " %");

end;

println("Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ", 100*std(testAccuracies));
println("Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ", 100*std(testF1));