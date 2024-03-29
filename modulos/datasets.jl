using FileIO;
using DelimitedFiles;
using Statistics: mean, std;
using Random: randperm;

# =============================================================================
# datasets.jl -> Funciones útiles aplicables a un dataset:
#   - oneHotEncoding
#   - normalizar:
#       -calculateMinMaxNormalizationParameters
#       -normalizeMinMax!
#       -calculateZeroMeanNormalizationParameters
#       -normalizeZeroMean!
#   - holdOut (sobrecargada 2 y 3 params)
#   - validacion cruzada(añadir en un futuro) -> FALTAS TU
#   - classifyOutputs
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
# Extraemos las clases distintas con unique
oneHotEncoding(feature::Array{Any,1}) = oneHotEncoding(feature::Array{Any,1}, unique(feature));
#  En este caso, el propio vector ya está codificado
oneHotEncoding(feature::Array{Bool,1}) = feature;

# =============================================================================
# Funciones útiles para la normalización de un dataset
# =============================================================================


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
    end;
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

# =============================================================================
# Obtener los conjuntos discuntos adecuados (holdOut)
# =============================================================================

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


# =============================================================================
# classifyOutputs -> Pasar de una salida en float64 a bool
# =============================================================================

function classifyOutputs(outputs::Array{Float64,2}; dataInRows::Bool=true, threshold::Float64=0.5)
    numOutputs = size(outputs, dataInRows ? 2 : 1);
    @assert(numOutputs!=2)
    if numOutputs==1
        return convert(Array{Bool,2}, outputs.>=threshold);
    else
        # Miramos donde esta el valor mayor de cada instancia con la funcion findmax
        (_,indicesMaxEachInstance) = findmax(outputs, dims= dataInRows ? 2 : 1);
        # Creamos la matriz de valores booleanos con valores inicialmente a false y asignamos esos indices a true
        outputsBoolean = Array{Bool,2}(falses(size(outputs)));
        outputsBoolean[indicesMaxEachInstance] .= true;
        # Comprobamos que efectivamente cada patron solo este clasificado en una clase
        @assert(all(sum(outputsBoolean, dims=dataInRows ? 2 : 1).==1));
        return outputsBoolean;
    end;
end;
