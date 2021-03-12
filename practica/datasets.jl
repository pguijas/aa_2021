# =============================================================================
# Funciones útiles para tratar con el dataset
# =============================================================================

using FileIO;
using DelimitedFiles;



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


#
# Función auxiliar para normalizar los datos mediante minimo-maximo.
#
# @arguments
#   feature: los inputs.
#
# @return: outputs en one-hot-encoding
#
calculateMinMaxNormalizationParameters(dataset::Array{Float64,2}; dataInRows=true) =
    ( minimum(dataset, dims=(dataInRows ? 1 : 2)), maximum(dataset, dims=(dataInRows ? 1 : 2)) );


#
# Función auxiliar para normalizar los datos mediante media cero.
#
# @arguments
#   feature: los inputs.
#
# @return: outputs en one-hot-encoding
#
calculateZeroMeanNormalizationParameters(dataset::Array{Float64,2}; dataInRows=true) =
    ( mean(dataset, dims=(dataInRows ? 1 : 2)), std(dataset, dims=(dataInRows ? 1 : 2)) );

# IMPORTANTE PARA ENTENDER LO DE ABAJO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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




#dataset = readdlm("../scripts/intro/iris.data",',');
