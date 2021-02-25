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
# @args:
#   topology: array que tiene el num de capas, neuronas/capa y funciones
#       de activación por capa.
#   inputs: matriz con los inputs del problema.
#   trargets: matriz con los valores de salida deseados.
#   stopping_cond: condiciones de parada para el entrenamiento.
#
# @return ann: red ya entrenada
#
function newAnn(topology::Array, inputs::Array, targets::Array, stoping_cond::Any)

    # creanos una rna
    ann = Chain();

    # miramos cuantas entradas tiene nuestro problema
    numInputsLayer = size(inputs,2)
    # el vector topology tiene el num de capas, neuronas/capa y la funcion
    # de activación de cada capa. Si no hay neuronas ocultas esta vacío.
    for tuple = topology
        numOutputsLayer = tuple[1]
        activation_function = tuple[2]
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, activation_function));
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

    return ann;

end

# función de activación reLu de teoría
relu(x) = max(0, x)

# creo una rna con 2 capas ocultas, 4 neuronas en la capa de entrada (pq en el
# dataset de iris tenemos 4 valores para categorizar) y 5 en la primera capa
# oculta, y 8 en la segunda capa oculta. Además de la capa de salida con 3
# neuronas (3 clases). Además, a cada capa oculta se le pasa su función de
# transferencia no tiene por qué ser la funcion Sigmoid, podemos usar relu o
# cualquier otra funcion.
ann = newAnn([(5, σ); (8, relu)], inputs, targets, [])
# neurona sin capas ocultas
ann2 = newAnn([], inputs, targets, [])

# si entrena 1 iteracción es que funciona :)
loss(x, y) = Losses.crossentropy(ann(x), y)
Flux.train!(loss, params(ann), [(inputs', targets')], ADAM(0.01));
Flux.train!(loss, params(ann2), [(inputs', targets')], ADAM(0.01));

# TO-DO:
#   entrenar la rna dentro del bucle
