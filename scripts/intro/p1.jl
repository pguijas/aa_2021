using DelimitedFiles
using Flux
using Flux.Losses
include("intro_v2.jl")

#
#  Esta función debe recibir la topología (número de capas y neuronas y
#   funciones de activación en cada capa), conjunto de entrenamiento, y
#   condiciones de parada. Una vez creada la RNA, esta función debería
#   implementar un bucle en el que se entrene la RNA con el conjunto de
#   entrenamiento pasado como parámetro hasta que se cumple uno de los
#   criterios de parada pasados como parámetros.La función debería devolver la
#   RNA entrenada.
#
# Args:
#   topology: array que tiene el num de capas, neuronas/capa y funciones
#       de activación por capa.
#   inputs: matriz con los inputs del problema.
#   trargets: matriz con los valores de salida deseados.
#   stopping_cond: condiciones de parada para el entrenamiento.
#
function newAnn(topology::Array{Int64,1}, inputs::Array, targets::Array, stoping_cond::Any)

    # creanos una rna
    ann = Chain();

    # miramos cuantas entradas tiene nuestro problema
    numInputsLayer = size(inputs,1)

    # el vector topology tiene el num de capas, neuronas/capa y la funcion
    # de activación de cada capa. Si no hay neuronas ocultas esta vacío.
    for numOutputsLayer = topology
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, σ));
        numInputsLayer = numOutputsLayer;
    end

    # miramos las clases de salida que tiene nuestro problema
    output_layers = size(targets,2)

    # si tiene más de dos clases usamos una neurona de salida por clase
    if (output_layers > 1)
        ann = Chain(ann..., Dense(numInputsLayer, output_layers, identity));
        ann = Chain(ann..., softmax);

    # si solo tenemos dos clases, usamos una rollo positivo/negativo
    else
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    end

    # entrenar la rnna
    #loss(x, y) = Losses.crossentropy(ann(x), y);
    #Flux.train!(loss, params(ann), [(inputs', targets')], ADAM(0.01));

    return ann;

end

# creo una rna con 2 capas ocultas, 5 neuronas en la primera y 8 en la segunda.
ann = newAnn([5,8], inputs, targets, [])


# TO-DO list:
#   en topology tb hay que meter funciones de activacion
#   entrenar la rna.
