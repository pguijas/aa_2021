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

# Sobrecargamos la funcion oneHotEncoding por si acaso pasan un vector de valores booleanos
#  En este caso, el propio vector ya está codificado
# Cuando se llame a la funcion oneHotEncoding, según el tipo del argumento pasado, Julia realizará
#  la llamada a la función de arriba o esta
oneHotEncoding(feature::Array{Bool,1}) = feature;


#dataset = readdlm("../scripts/intro/iris.data",',');

# Preparamos las entradas
#inputs = dataset[:,1:4];
# Con cualquiera de estas 3 maneras podemos convertir la matriz de entradas de tipo Array{Any,2} en Array{Float64,2}, si los valores son numéricos:
#inputs = Float64.(inputs);
#inputs = convert(Array{Float64,2},inputs);
#inputs = [Float64(x) for x in inputs];
#println("Tamaño de la matriz de entradas: ", size(inputs,1), "x", size(inputs,2), " de tipo ", typeof(inputs));

# Preparamos las salidas deseadas codificándolas puesto que son categóricas
#targets = dataset[:,5];
#println("Longitud del vector de salidas deseadas antes de codificar: ", length(targets), " de tipo ", typeof(targets));
#targets = oneHotEncoding(targets);
#println("Tamaño de la matriz de salidas deseadas despues de codificar: ", size(targets,1), "x", size(targets,2), " de tipo ", typeof(targets));

# Comprobamos que ambas matrices tienen el mismo número de filas
#@assert (size(inputs,1)==size(targets,1)) "Las matrices de entradas y salidas deseadas no tienen el mismo numero de filas"
