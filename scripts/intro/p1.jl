using DelimitedFiles
using Flux: Chain, Dense, σ, softmax, crossentropy, params, ADAM, train!, binarycrossentropy
include("p0.jl")

# DUDAS:
#   el global es solo si no esta dentro de una func no?????
#   los parametros de parada en un array no es demasiado engorroso?¿?, yo veo bien que sean un parametro
#   en vez de trasponer cosas y no tenerlo ya en la forma en la que trabaja julia, mejor poner ya a patrones x columna en vez de x fila y trabajar con eso

# Curiosidades
#   ¿sabeis xq julia trabaja con patrones x columna y no por fila?
#       porque el orden secuencial de una matriz a[i] recorre las columnas y cuando las acaba pasa a la siguente columna de la siguiente fila


# Cosas:
#   falta la preciosion: falta pasar los resultados de la ann(inputs) (Array{Float32}) a (Array{Bool)
#   falta la normalización


#Para trasponer y que quede con el tipo adecuado:
#convert(Array{Bool}, targets')

# Precondición: salida en columnas columna, patrones en filas-> ojo a si trasponer o no
function precision(targets::Array, outputs::Array, is_transpose::Bool)
    #Comprobamos que sean del mismo tamaño
    @assert (size(targets) == size(outputs))
    #if size(targets,2)<size(targets,1)
    #    println("Hermano, te has olvidado de trasponer la matriz")
    #end
    if is_transpose
        i = 2
        j = 1
    else
        i = 1
        j = 2
    end
    n_patrones=size(targets,i)
    #Comparamos arrays
    comp= targets.==outputs
    #Aquí lo que hacemos es mirar que coincidan las salidas para todas las calses, si hay algun 0 no coincidirá :(
    correctos = all(comp, dims=j)
    #Calculamos precision
    return count(correctos)/n_patrones
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
function newAnn!(topology::Array, inputs::Array, targets::Array, stoping_cond::Any)
    # ann y numInputsLayer van a ser referenciadas fuera del bucle, entonces
    # hace falta ponerle "global"
    global ann, numInputsLayer

    # creamos una rna
    ann = Chain();
    # miramos cuantas entradas tiene nuestro problema
    numInputsLayer = size(inputs,1)
    # le pasamos a la función el número de capas junto con sus funciones de
    # activación
    for tuple = topology
        numOuputsLayer = tuple[1]
        activation_function = tuple[2]
        ann = Chain(ann..., Dense(numInputsLayer, numOuputsLayer, activation_function));
        numInputsLayer = numOuputsLayer;
    end
    # miramos las clases de salida que tiene nuestro problema
    output_layers = size(targets,1)

    # si tiene más de dos clases usamos una neurona de salida por clase
    if (output_layers > 1)
        ann = Chain(ann..., Dense(numInputsLayer, output_layers, identity));
        ann = Chain(ann..., softmax);
    # si solo tenemos dos clases, usamos una rollo positivo/negativo
    else
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    end

    # nuestra función de loss (crossentropy)
    loss(x, y) = (size(y,1) == 1) ? binarycrossentropy(ann(x),y) : crossentropy(ann(x),y);
    # pesos y bías de la rna
    ps = params(ann)
    # gradiente descentiente stochastic ADAM (optimizador)
    opt = ADAM(0.01)
    # condiciones de parada
    iterations = stoping_conditions[1]
    max_accurracy = stoping_conditions[2]
    min_error_value = stoping_conditions[3]
    # esto es para que saque por pantalla el error de crossentropy en cada it.
    evalcb = () -> @show(loss(inputs, targets))

    # comenzamos el bucle para entrenar la rna
    best_ann = ann
    best_loss = loss(inputs, targets)
    actual_loss = best_loss
    while (true)
        mod_err = actual_loss
        train!(loss, ps, [(inputs, targets)], opt, cb = evalcb);
        actual_loss = loss(inputs, targets)
        # si el error de loss de la última iteracción es mejor que el mejor que
        # hemos error registrado, actualizamos nuestra red
        if (best_loss > actual_loss)
            best_loss = actual_loss
            best_ann = ann
        end
        mod_err -= actual_loss
        @show(actual_loss)
        @show(iterations)
        @show(mod_err)
        iterations-=1
        # si nos pasamos de iteraciones, o la precisión es superior a un máximo
        # prefijado, o la modificación del error es demasiado baja, salimos
        if ((iterations < 0) || (actual_loss < max_accurracy) || (mod_err < min_error_value))
            break
        end
    end
    ann = best_ann
    @show(best_loss)
    @show(loss(inputs, targets))
end


# función auxiliar que normaliza entre max y min, para poder vectorizar
m(v, max, min) = (v - min)/(max - min)


# Esta función se encarga de normalizar las entradas para una rna. Utiliza la
#   normalización entre máximo y mínimo.
#
# @args:
#   inputs: las entradas de la rna.
#   is_transpose: si la matriz es traspuesta o no.
#
# @return: matriz normalizada.
#
function max_min_norm!(inputs::Array, is_transpose::Bool)
    global inputs
    cols = (is_transpose) ? size(inputs,1) : size(inputs,2)
    result = []
    i = 1
    while (i<=cols) # hay que vectorizar esto, pero no se me ocurre aun como
        row = inputs[i, :]
        max = maximum(row)
        min = minimum(row)
        if result == []
            result = transpose(m.(row,max,min))
        else
            result = [result; transpose(m.(row,max,min))]
        end
        i+=1
    end
    inputs = result
end


max_min_norm!(inputs,true);
@show(inputs);

# función de activación reLu de teoría
relu(x) = max(0, x)

# número máximo de iteraciones, accurracy máximo, modificación del error
stoping_conditions = [200, 0.9, 0]

# creo una rna con 2 capas ocultas, 4 neuronas en la capa de entrada (pq en el
# dataset de iris tenemos 4 valores para categorizar) y 5 en la primera capa
# oculta, y 8 en la segunda capa oculta. Además de la capa de salida con 3
# neuronas (3 clases). Además, a cada capa oculta se le pasa su función de
# transferencia no tiene por qué ser la funcion Sigmoid, podemos usar relu o
# cualquier otra funcion.
newAnn!([(5, σ); (8, relu)], inputs, targets, stoping_conditions)
# neurona sin capas ocultas
#ann2 = newAnn([], inputs, targets, [])


x = [1  0  0; 1  0  0; 1  0  0; 1  0  0]
y = [0  0  0; 1  0  0; 1  0  0; 1  0  0]
@show(precision(x,y,false))
x = convert(Array{Bool,2}, x')
y = convert(Array{Bool,2}, y')

@show(precision(x,y,true))



# TO-DO:
#   acabar p2
