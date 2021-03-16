# =============================================================================
# rna.jl -> Funciones útiles para crear y entrenar RNAs:
#   - buildClassANN
#   - trainClassANN
# =============================================================================

using Flux: Chain, Dense, σ, softmax, Losses, params, ADAM, train!;

#
# Función que crea una rna según la topología que se le indica.
#
# @arguments
#   numInputs: número de capas de entrada.
#   topology: topología de las capas ocultas.
#   numOutputs: número de capas de salida.
#
# @return: rna
#
function buildClassANN(numInputs::Int64, topology::Array{Int64,1}, numOutputs::Int64)
    ann=Chain();
    numInputsLayer = numInputs;
    for numOutputLayers = topology
        ann = Chain(ann..., Dense(numInputsLayer, numOutputLayers, σ));
        numInputsLayer = numOutputLayers;
    end;
    #Dependiendo del nº de salidas añadimos las capas de salida adecuadas
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);
    end;
    return ann;
end;


#
# Funcion para entrenar RR.NN.AA. con conjuntos de entrenamiento, validacion y
#   test. Calcular errores en el conjunto de validacion, y para el entrenamiento
#   si es necesario.
#
# @arguments
#   topology: topología de las capas ocultas.
#   trainingInputs: las entradas que representan el conjunto de entrenamiento.
#   trainingTargets: las salidas deseadas del conjunto de entrenamiento.
#   validationInputs: inputs para realizar la validación del dataset.
#   validationTargets: salidas deseadas para realizar la validación del dataset.
#   testInputs: inputs para el conjunto de test.
#   testTargets: salidas deseadas para el conjunto de test.
#   maxEpochs: número máximo de iteraciones.
#   minLoss: mínimo error de entrenamiento.
#   learningRate: tasa de aprendizaje.
#   maxEpochsVal: número máximo de iteraciones sin mejorar en el conjunto de validacion.
#   showText: saca por pantalla el texto si es cierto
#
# @return: rna entrenada. ¿¿home devolverá mais cousiñas non??? mucha mariconada esta inspiración en los javadocs
#
function trainClassANN(topology::Array{Int64,1}, trainingInputs::Array{Float64,2}, trainingTargets::Array{Bool,2},
    validationInputs::Array{Float64,2}, validationTargets::Array{Bool,2}, testInputs::Array{Float64,2}, testTargets::Array{Bool,2};
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
    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : crossentropy(ann(x),y);
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
        trainingLoss = loss(trainingInputs', trainingTargets');
        validationLoss = loss(validationInputs', validationTargets');
        testLoss = loss(testInputs', testTargets');
        # Calculamos la salida de la RNA en entrenamiento y test. Para ello hay que pasar la matriz de entradas traspuesta (cada patron en una columna). La matriz de salidas tiene un patron en cada columna
        trainingOutputs = ann(trainingInputs');
        validationOutputs = ann(validationInputs');
        testOutputs = ann(testInputs');
        # Para calcular la precision, ponemos 2 opciones aqui equivalentes:
        #  Pasar las matrices con los datos en las columnas. La matriz de salidas ya tiene un patron en cada columna
        trainingAcc = accuracy(trainingOutputs, Array{Bool,2}(trainingTargets'); dataInRows=false);
        validationAcc = accuracy(validationOutputs, Array{Bool,2}(validationTargets'); dataInRows=false);
        testAcc = accuracy(testOutputs, Array{Bool,2}(testTargets'); dataInRows=false);
        #  Pasar las matrices con los datos en las filas. Hay que trasponer la matriz de salidas de la RNA, puesto que cada dato esta en una fila
        trainingAcc = accuracy(Array{Float64,2}(trainingOutputs'), trainingTargets;   dataInRows=true);
        validationAcc = accuracy(Array{Float64,2}(validationOutputs'), validationTargets; dataInRows=true);
        testAcc = accuracy(Array{Float64,2}(testOutputs'), testTargets; dataInRows=true);
        # Mostramos por pantalla el resultado de este ciclo de entrenamiento si nos lo han indicado
        if showText
            #println("Epoch ", numEpoch, ": Training loss: ", trainingLoss, ", accuracy: ", 100*trainingAcc, " % - Validation loss: ", validationLoss, ", accuracy: ", 100*validationAcc, " % - Test loss: ", testLoss, ", accuracy: ", 100*testAcc, " %");
            println("Epoch ", numEpoch);
            println("\t Training loss: ", trainingLoss, ", accuracy: ", 100*trainingAcc);
            println("\t Validation loss: ", validationLoss, ", accuracy: ", 100*validationAcc);
            println("\t Test loss: ", testLoss, ", accuracy: ", 100*testAcc);
        end;
        return (trainingLoss, trainingAcc, validationLoss, validationAcc, testLoss, testAcc)
    end;

    # Calculamos las metricas para el ciclo 0 (sin entrenar nada)
    (trainingLoss, trainingAccuracy, validationLoss, validationAccuracy, testLoss, testAccuracy) = calculateMetrics();
    #  y almacenamos los valores de loss y precision en este ciclo
    println("PENE")
    println(trainingLoss)
    println(typeof(trainingLosses))
    println(typeof(trainingLoss))
    push!(trainingLosses, trainingLoss);
    println("Culo")
    push!(trainingAccuracies, trainingAccuracy);
    push!(validationLosses, validationLoss);
    push!(validationAccuracies, validationAccuracy);
    push!(testLosses, testLoss);
    push!(testAccuracies, testAccuracy);

    # Numero de ciclos sin mejorar el error de validacion y el mejor error de validation encontrado hasta el momento
    numEpochsValidation = 0; bestValidationLoss = validationLoss;
    # Cual es la mejor ann que se ha conseguido
    bestANN = deepcopy(ann);

    # Entrenamos hasta que se cumpla una condicion de parada
    while (numEpoch<maxEpochs) && (trainingLoss>minLoss) && (numEpochsValidation<maxEpochsVal)
        # Entrenamos 1 ciclo. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        train!(loss, params(ann), [(trainingInputs', trainingTargets')], ADAM(learningRate));
        # Aumentamos el numero de ciclo en 1
        numEpoch += 1;
        # Calculamos las metricas en este ciclo
        (trainingLoss, trainingAccuracy, validationLoss, validationAccuracy, testLoss, testAccuracy) = calculateMetrics();
        #  y almacenamos los valores de loss y precision en este ciclo
        push!(trainingLosses, trainingLoss);
        push!(trainingAccuracies, trainingAccuracy);
        push!(validationLosses, validationLoss);
        push!(validationAccuracies, validationAccuracy);
        push!(testLosses, testLoss);
        push!(testAccuracies,testAccuracy);
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
