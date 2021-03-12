# =============================================================================
# Funciones útiles para crear y entrenar una rna
# =============================================================================


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
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);
    end;
    return ann;
end;


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
        # println("Epoch ", numEpoch, ": loss: ", trainingLoss, ", accuracy: ", 100*acc, " %");
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
