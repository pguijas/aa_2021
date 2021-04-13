include("bondad.jl")
include("rna.jl")

using ScikitLearn;
@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier
using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    indices = repeat(1:k, Int64(ceil(N/k)));
    indices = indices[1:N];
    shuffle!(indices);
    return indices;
end;

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

confusionMatrix(outputs::Array{Float64,2}, targets::Array{Bool,2}; weighted::Bool=true) = confusionMatrix(classifyOutputs(outputs), targets; weighted=weighted);
confusionMatrix(outputs::Array{Float64,1}, targets::Array{Bool,1}; threshold::Float64=0.5) = confusionMatrix(Array{Bool,1}(outputs.>=threshold), targets);
confusionMatrix(outputs::Array{Float64,2}, targets::Array{Bool,2}; weighted::Bool=true) = confusionMatrix(classifyOutputs(outputs), targets; weighted=weighted);
confusionMatrix(outputs::Array{Float32,2}, targets::Array{Bool,2}; weighted::Bool=true) = confusionMatrix(convert(Array{Float64,2}, outputs), targets; weighted=weighted);


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

function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, inputs::Array{Float64,2}, targets::Array{Any,1}, numFolds::Int64)

    # Comprobamos que el numero de patrones coincide
    @assert(size(inputs,1)==length(targets));

    # Que clases de salida tenemos
    # Es importante calcular esto primero porque se va a realizar codificacion one-hot-encoding varias veces, y el orden de las clases deberia ser el mismo siempre
    classes = unique(targets);

    # Primero codificamos las salidas deseadas en caso de entrenar RR.NN.AA.
    if modelType==:ANN
        targets = oneHotEncoding(targets, classes);
    end;

    # Creamos los indices de crossvalidation
    crossValidationIndices = crossvalidation(size(inputs,1), numFolds);

    # Creamos los vectores para las metricas que se vayan a usar
    # En este caso, solo voy a usar precision y F1, en otro problema podrían ser distintas
    testAccuracies = Array{Float64,1}(undef, numFolds);
    testF1         = Array{Float64,1}(undef, numFolds);

    # Para cada fold, entrenamos
    for numFold in 1:numFolds

        # Si vamos a usar unos de estos 3 modelos
        if (modelType==:SVM) || (modelType==:DecisionTree) || (modelType==:kNN)

            # Dividimos los datos en entrenamiento y test
            trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
            testInputs        = inputs[crossValidationIndices.==numFold,:];
            trainingTargets   = targets[crossValidationIndices.!=numFold];
            testTargets       = targets[crossValidationIndices.==numFold];

            if modelType==:SVM
                model = SVC(kernel=modelHyperparameters["kernel"], degree=modelHyperparameters["kernelDegree"], gamma=modelHyperparameters["kernelGamma"], C=modelHyperparameters["C"]);
            elseif modelType==:DecisionTree
                model = DecisionTreeClassifier(max_depth=modelHyperparameters["maxDepth"], random_state=1);
            elseif modelType==:kNN
                model = KNeighborsClassifier(modelHyperparameters["numNeighbors"]);
            end;

            # Entrenamos el modelo con el conjunto de entrenamiento
            model = fit!(model, trainingInputs, trainingTargets);

            # Pasamos el conjunto de test
            testOutputs = convert(Array{Bool,1},predict(model, testInputs));
            testTargets = convert(Array{Float64,1},testTargets);
            # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
            (acc, _, _, _, _, _, F1, _) = confusionMatrix(testTargets, testOutputs);

        else
            # Vamos a usar RR.NN.AA.
            @assert(modelType==:ANN);

            # Dividimos los datos en entrenamiento y test
            trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
            testInputs        = inputs[crossValidationIndices.==numFold,:];
            trainingTargets   = targets[crossValidationIndices.!=numFold,:];
            testTargets       = targets[crossValidationIndices.==numFold,:];

            # Como el entrenamiento de RR.NN.AA. es no determinístico, hay que entrenar varias veces, y
            #  se crean vectores adicionales para almacenar las metricas para cada entrenamiento
            testAccuraciesEachRepetition = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);
            testF1EachRepetition         = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);

            # Se entrena las veces que se haya indicado
            for numTraining in 1:modelHyperparameters["numExecutions"]

                if modelHyperparameters["validationRatio"]>0

                    # Para el caso de entrenar una RNA con conjunto de validacion, hacemos una división adicional:
                    #  dividimos el conjunto de entrenamiento en entrenamiento+validacion
                    #  Para ello, hacemos un hold out
                    (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), modelHyperparameters["validationRatio"]*size(trainingInputs,1)/size(inputs,1));
                    # Con estos indices, se pueden crear los vectores finales que vamos a usar para entrenar una RNA

                    # Entrenamos la RNA, teniendo cuidado de codificar las salidas deseadas correctamente
                    ann, = trainClassANN(modelHyperparameters["topology"],
                        trainingInputs[trainingIndices,:],   trainingTargets[trainingIndices,:],
                        trainingInputs[validationIndices,:], trainingTargets[validationIndices,:],
                        testInputs,                          testTargets;
                        maxEpochs=modelHyperparameters["maxEpochs"], learningRate=modelHyperparameters["learningRate"], maxEpochsVal=modelHyperparameters["maxEpochsVal"]);

                else

                    # Si no se desea usar conjunto de validacion, se entrena unicamente con conjuntos de entrenamiento y test,
                    #  teniendo cuidado de codificar las salidas deseadas correctamente
                    ann, = trainClassANN(modelHyperparameters["topology"],
                        trainingInputs, trainingTargets,
                        testInputs,     testTargets;
                        maxEpochs=modelHyperparameters["maxEpochs"], learningRate=modelHyperparameters["learningRate"]);

                end;

                # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
                (testAccuraciesEachRepetition[numTraining], _, _, _, _, _, testF1EachRepetition[numTraining], _) = confusionMatrix(collect(ann(testInputs')'), testTargets);

            end;

            # Calculamos el valor promedio de todos los entrenamientos de este fold
            acc = mean(testAccuraciesEachRepetition);
            F1  = mean(testF1EachRepetition);

        end;

        # Almacenamos las 2 metricas que usamos en este problema
        testAccuracies[numFold] = acc;
        testF1[numFold]         = F1;

        println("Results in test in fold ", numFold, "/", numFolds, ": accuracy: ", 100*testAccuracies[numFold], " %, F1: ", 100*testF1[numFold], " %");

    end; # for numFold in 1:numFolds

    println(modelType, ": Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ", 100*std(testAccuracies));
    println(modelType, ": Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ", 100*std(testF1));

    return (mean(testAccuracies), std(testAccuracies), mean(testF1), std(testF1));

end;
