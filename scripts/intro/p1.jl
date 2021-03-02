using DelimitedFiles
using Flux
using Flux.Losses
using Flux: @epochs
include("intro_v2.jl")


#
#  Esta función auxiliar para añadir una capa oculta a la red neuronal
#
# @args:
#   hidden_neurons: número de neuronas de la capa oculta que estamos creando.
#   activation_function: función de activación para la capa.
#
function load_hidden_layers(hidden_neurons::Any, activation_function::Any)
    # ann y numInputsLayer van a ser referenciadas fuera del bucle, entonces
    # hace falta ponerle "global"
    global ann, numInputsLayer
    # añade una capa oculta a la rna que ya teníamos
    ann = Chain(ann..., Dense(numInputsLayer, hidden_neurons, activation_function));
    # seguimos iterando, añadido el último valor de neuronas para ponerlo en la
    # siguiente capa y que puedan conectarse.
    numInputsLayer = hidden_neurons;
end


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
#   stopping_cond: condiciones de parada para el entrenamiento, es un Array
#       en el que el primer elemento es el numero de iteraciones, el segundo
#       error de entrenamiento de parada y el último la mínima modificación
#       del error de entrenamiento.
#
# @return ann: red ya entrenada
#
function newAnn(num_layers::Array, functions::Array, inputs::Array, targets::Array, stoping_cond::Any)
    # ann y numInputsLayer van a ser referenciadas fuera del bucle, entonces
    # hace falta ponerle "global"
    global ann, numInputsLayer

    # creamos una rna
    ann = Chain();
    # miramos cuantas entradas tiene nuestro problema
    numInputsLayer = size(inputs,2)
    # si el número de capas es diferente al de neuronas de activación no podemos
    # entrenar.
    @assert (size(num_layers) == size(functions))
    # le pasamos a la función el número de capas junto con sus funciones de
    # activación
    load_hidden_layers.(num_layers,functions)
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

    # nuestra función de loss (crossentropy)
    loss(x, y) = Losses.crossentropy(ann(x), y)
    # pesos y bías de la rna
    ps = params(ann)
    # condiciones de parada
    iterations = stoping_conditions[1]
    max_accurracy = stoping_conditions[2]
    min_error_value = stoping_conditions[3]

    # entrena la rna "iterations" veces o "iterations" epochs según los de Flux          # aquí va el valor de accurracy de la rna 
    @epochs iterations Flux.train!(loss, ps, [(inputs', targets')], ADAM(0.01), cb = function () 0.9 > 0.9 && Flux.stop() end);

    return ann;

end


# función de activación reLu de teoría
relu(x) = max(0, x)

# número máximo de iteraciones, accurracy máximo, modificación del error
stoping_conditions = [20, 0.9, 0]

# creo una rna con 2 capas ocultas, 4 neuronas en la capa de entrada (pq en el
# dataset de iris tenemos 4 valores para categorizar) y 5 en la primera capa
# oculta, y 8 en la segunda capa oculta. Además de la capa de salida con 3
# neuronas (3 clases). Además, a cada capa oculta se le pasa su función de
# transferencia no tiene por qué ser la funcion Sigmoid, podemos usar relu o
# cualquier otra funcion.
ann = newAnn([5, 8], [σ, relu], inputs, targets, stoping_conditions)
# neurona sin capas ocultas
#ann2 = newAnn([], inputs, targets, [])

# si entrena 1 iteracción es que funciona :)
#Flux.train!(loss, params(ann2), [(inputs', targets')], ADAM(0.01))


# TO-DO:
#   falta definir una función que haga earlystopping porque el error o loss de
#   entrenamiento es lo suficientemente bueno o la modificación del error de
#   entrenamiento es inferior a un valor prefijado
