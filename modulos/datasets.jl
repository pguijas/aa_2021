# =============================================================================
# datasets.jl -> Funciones útiles aplicables a un dataset:
#   - oneHotEncoding
#   - normalizar:
#       -get norm params
#       -normalize
#   - holdOut (sobrecargada 2 y 3 params)
#   - precision no pinta nada aquí
# =============================================================================


using FileIO;
using DelimitedFiles;
using Statistics: mean, std;
using Random: randperm;

# =============================================================================
# Transformar dataset en formato adecuado (oneHotEncoding)
# =============================================================================

#
# Esta función sirve para normalizar las salidas deseadas del dataset para un
#   problema de clasificacion, tanto como si es binario o multiclase.
#
# @arguments
#   feature: Array con los outputs deseados
#
# @return: outputs en one-hot-encoding
#
function oneHotEncoding(feature::Array{Any,1})::Array{Bool,2}
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
    return oneHot
end;
# en caso de que ya sea un array de bools con una sola salida, lo devolvemos
oneHotEncoding(feature::Array{Bool,1}) = feature;


# =============================================================================
# Funciones útiles para la normalización de un dataset
# =============================================================================


#
# Función que calcula la precision entre dos arrays de booleanos.
#
# @arguments
#   outputs: salidas.
#   targets: salidas deseadas.
#
# @return: similitud entre los arrays.
#
accuracy(outputs::Array{Bool,1}, targets::Array{Bool,1}) = mean(outputs.==targets);

#
# Función que calcula la precision entre dos arrays multiclase.
#
# @arguments
#   outputs: salidas.
#   targets: salidas deseadas.
#
# @return: similitud entre los arrays.
#
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

#
# Función que calcula la precision entre dos arrays numéricos.
#
# @arguments
#   outputs: salidas.
#   targets: salidas deseadas.
#
# @return: similitud entre los arrays.
#
accuracy(outputs::Array{Float64,1}, targets::Array{Bool,1}; threshold::Float64=0.5) = accuracy(Array{Bool,1}(outputs.>=threshold), targets);

#
# Función que calcula la precision entre dos matrices numéricas.
#
# @arguments
#   outputs: salidas.
#   targets: salidas deseadas.
#
# @return: similitud de las matrices.
#
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
# Función auxiliar para calcular el máximo y mínimo de cada atributo.
#
# @arguments
#   dataset: los inputs.
#   dataInRows: si esta traspuesta o no.
#
# @return: array con el máximo y mínimo de cada atributo.
#
calculateMinMaxNormalizationParameters(dataset::Array{Float64,2}; dataInRows=true) =
    ( minimum(dataset, dims=(dataInRows ? 1 : 2)), maximum(dataset, dims=(dataInRows ? 1 : 2)) );

#
# Función AUXILIAR para normalizar los inputs entre máximo y mínimo.
#
# @arguments
#   dataset: los inputs.
#   normalizationParameters: array con el máximo y mínimo de cada atributo.
#   dataInRows: si esta traspuesta o no.
#
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

#
# Función que normaliza los inputs entre máximo y mínimo (es a la que llamamos).
#
# @arguments
#   dataset: los inputs.
#   dataInRows: si la matriz está traspuesta o no.
#
normalizeMinMax!(dataset::Array{Float64,2}; dataInRows=true) = normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset; dataInRows=dataInRows); dataInRows=dataInRows);

#
# Función auxiliar para calcular la media y la desviacion típica de cada atributo.
#
# @arguments
#   dataset: los inputs.
#   dataInRows: si esta traspuesta o no.
#
# @return: array con la media y la desviacion típica de cada atributo.
#
calculateZeroMeanNormalizationParameters(dataset::Array{Float64,2}; dataInRows=true) =
( mean(dataset, dims=(dataInRows ? 1 : 2)), std(dataset, dims=(dataInRows ? 1 : 2)) );

#
# Función AUXILIAR para normalizar los inputs con media cero.
#
# @arguments
#   dataset: los inputs.
#   normalizationParameters: array con la media y la desviacion típica de cada atributo.
#   dataInRows: si esta traspuesta o no.
#
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

#
# Función que normaliza los inputs con media 0 (es a la que llamamos).
#
# @arguments
#   dataset: los inputs.
#   dataInRows: si la matriz está traspuesta o no.
#
normalizeZeroMean!(dataset::Array{Float64,2}; dataInRows=true) = normalizeZeroMean!(dataset, calculateZeroMeanNormalizationParameters(dataset; dataInRows=dataInRows); dataInRows=dataInRows);



#
# Función que divide el conjunto de entrenamiento en entrenamiento y test.
#
# @arguments
#   N: número de patrones.
#   P: porcentaje de división test/entrenamiento.
#
# @return nuevos conjuntos de entrenamiento y test
#
function holdOut(N::Int, P::Float64)
    @assert ((P>=0.) & (P<=1.));
    indices = randperm(N);
    numTrainingInstances = Int(round(N*(1-P)));
    return (indices[1:numTrainingInstances], indices[numTrainingInstances+1:end]);
end

#
# Función que divide el conjunto de entrenamiento en entrenamiento, test y validación.
#
# @arguments
#   N: número de patrones.
#   Ptest: porcentaje de división test/entrenamiento.
#   Pval: porcentaje de división validación/entrenamiento.
#
# @return nuevos conjuntos de entrenamiento, test y validacion.
#
function holdOut(N::Int, Ptest::Float64, Pval::Float64)
    @assert ((Pval>=0.) & (Pval<=1.));
    @assert ((Ptest>=0.) & (Ptest<=1.));
    @assert ((Pval+Ptest)<=1.);
    # Primero separamos en entrenamiento+validation y test
    (trainingValidationIndices, testIndices) = holdOut(N, Ptest);
    # Después separamos el conjunto de entrenamiento+validation
    (trainingIndices, validationIndices) = holdOut(length(trainingValidationIndices), Pval*N/length(trainingValidationIndices))
    return (trainingValidationIndices[trainingIndices], trainingValidationIndices[validationIndices], testIndices);
end;
