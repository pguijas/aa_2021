using Flux
using Flux: onehotbatch, onecold, crossentropy
using JLD2, FileIO
using Statistics: mean

#===============================================================================
COSAS:
    · en toFloatArray no me deja hacer un assert de una comp de ints:  MethodError: objects of type Int64 are not callable

===============================================================================#


include("modulos/dataset_DL.jl")

interval = 6;
(train_imgs, train_labels,
    test_imgs, test_labels) = getInputs("datasets", interval);

#=
for image in train_imgs
    display(image);
end;
for image in test_imgs
    display(image);
end;
=#

train_imgs = toFloatArray(train_imgs);
test_imgs = toFloatArray(test_imgs);


println("Tamaño de la matriz de entrenamiento: ", size(train_imgs))
println("Tamaño de la matriz de test:          ", size(test_imgs))



batch_size = 64;
gruposIndicesBatch = Iterators.partition(1:size(train_imgs,4), batch_size);
println("He creado ", length(gruposIndicesBatch), " grupos de indices para distribuir los patrones en batches");

train_set = [( train_imgs[:,:,:,indicesBatch], Array(onehotbatch(train_labels[indicesBatch], 0:2)) ) for indicesBatch in gruposIndicesBatch]

test_set = (test_imgs, onehotbatch(test_labels, 0:2));

train_imgs = nothing;
test_imgs = nothing;
GC.gc();

funcionTransferenciaCapasConvolucionales = relu;

# Definimos la red con la funcion Chain, que concatena distintas capas
modelo = Chain(
    Conv((3, 3), 3=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    Conv((3, 3), 32=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(10368, 3),
    softmax
);


# Vamos a probar la RNA y poner algunos datos de cada capa
# Usaremos como entrada varios patrones de un batch
numBatchCoger = 1; numImagenEnEseBatch = [12, 6];


entradaCapa = train_set[numBatchCoger][1][:,:,:,numImagenEnEseBatch];
numCapas = length(params(modelo));
println("La RNA tiene ", numCapas, " capas:");
for numCapa in 1:numCapas
    println("   Capa ", numCapa, ": ", modelo[numCapa]);
    # Le pasamos la entrada a esta capa
    global entradaCapa # Esta linea es necesaria porque la variable entradaCapa es global y se modifica en este bucle
    capa = modelo[numCapa];
    salidaCapa = capa(entradaCapa);
    println("      La salida de esta capa tiene dimension ", size(salidaCapa));
    entradaCapa = salidaCapa;
end

modelo(train_set[numBatchCoger][1][:,:,:,numImagenEnEseBatch]);

loss(x, y) = crossentropy(modelo(x), y)
accuracy(batch) = mean(onecold(modelo(batch[1])) .== onecold(batch[2]))

println("Ciclo 0: Precision en el conjunto de entrenamiento: ", 100*mean(accuracy.(train_set)), " %");

opt = ADAM(0.001);


println("Comenzando entrenamiento...")
mejorPrecision = -Inf;
criterioFin = false;
numCiclo = 0;
numCicloUltimaMejora = 0;
mejorModelo = nothing;

while (!criterioFin)

    global numCicloUltimaMejora, numCiclo, mejorPrecision, mejorModelo, criterioFin;

    Flux.train!(loss, params(modelo), train_set, opt);

    numCiclo += 1;

    precisionEntrenamiento = mean(accuracy.(train_set));
    println("Ciclo ", numCiclo, ": Precision en el conjunto de entrenamiento: ", 100*precisionEntrenamiento, " %");

    if (precisionEntrenamiento >= mejorPrecision)
        mejorPrecision = precisionEntrenamiento;
        precisionTest = accuracy(test_set);
        println("   Mejora en el conjunto de entrenamiento -> Precision en el conjunto de test: ", 100*precisionTest, " %");
        mejorModelo = deepcopy(modelo);
        numCicloUltimaMejora = numCiclo;
    end;

    if (numCiclo - numCicloUltimaMejora >= 5) && (opt.eta > 1e-6)
        opt.eta /= 10.0
        println("   No se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje a ", opt.eta);
        numCicloUltimaMejora = numCiclo;
    end;

    if (precisionEntrenamiento >= 0.999)
        println("   Se para el entenamiento por haber llegado a una precision de 99.9%")
        criterioFin = true;
    end;

    if (numCiclo - numCicloUltimaMejora >= 10)
        println("   Se para el entrenamiento por no haber mejorado la precision en el conjunto de entrenamiento durante 10 ciclos")
        criterioFin = true;
    end;
end;
