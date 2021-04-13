using FileIO;
using DelimitedFiles;
using ScikitLearn;
@sk_import svm: SVC
using Random
using Random:seed!

function trainClassSVM(trainingInputs::Array{Float64,2}, trainingTargets::Array{Bool,2},
    testInputs::Array{Float64,2}, testTargets::Array{Bool,2};
    kernel::String="rbf", kernelDegree::Int64=3, kernelGamma::Int64=2, C::Int64=1)

    # Comprobamos que el numero de filas (numero de patrones) coincide tanto en entrenamiento como en test
    @assert(size(trainingInputs,1)==size(trainingTargets,1));
    @assert(size(testInputs,1)==size(testTargets,1));
    # Comprobamos que el numero de columnas coincide en los grupos de entrenamiento y test
    @assert(size(trainingInputs,2)==size(testInputs,2));
    @assert(size(trainingTargets,2)==size(testTargets,2));

    # Creamos nuestra máquina de soporte vectorial
    model = SVC(kernel=kernel, degree=kernelDegree, gamma=kernelGamma, C=C);
    # Entrenamos el modelo con el conjunto de entrenamiento
    model = fit!(model, trainingInputs, trainingTargets);
    # Pasamos el conjunto de test
    testOutputs = predict(model, testInputs);

    return model;
end;
