include("modulos/bondad.jl")
include("modulos/graphics.jl")
include("modulos/datasets.jl")
include("modulos/rna.jl")
include("modulos/attributes_from_dataset.jl")
include("modulos/models_cross_validation.jl")

using Random;
using Flux;
using Random:seed!;

function topology_test(inputs::Array{Float64,2}, targets::Array{Bool,2}, topology::Array{Int64,1})

    learningRate = 0.01;
    numMaxEpochs = 1000;
    numFolds = 10;
    validationRatio = 0.2;
    maxEpochsVal = 30;
    numRepetitionsAANTraining = 50;

    testAccuracies = Array{Float64,1}(undef, numFolds);
    testF1         = Array{Float64,1}(undef, numFolds);

    crossValidationIndices = crossvalidation(size(inputs,1), numFolds);

    for numFold in 1:numFolds

        local trainingInputs, testInputs, trainingTargets, testTargets;
        trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
        testInputs        = inputs[crossValidationIndices.==numFold,:];
        trainingTargets   = targets[crossValidationIndices.!=numFold,:];
        testTargets       = targets[crossValidationIndices.==numFold,:];

        testAccuraciesEachRepetition = Array{Float64,1}(undef, numRepetitionsAANTraining);
        testF1EachRepetition         = Array{Float64,1}(undef, numRepetitionsAANTraining);

        for numTraining in 1:numRepetitionsAANTraining

            local trainingIndices, validationIndices;
            (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), validationRatio*size(trainingInputs,1)/size(inputs,1));

            local ann;
            (ann, training_acc, _, _, _, _, _) = trainClassANN(topology,
                trainingInputs[trainingIndices,:],   trainingTargets[trainingIndices,:],
                trainingInputs[validationIndices,:], trainingTargets[validationIndices,:],
                testInputs,                          testTargets;
                maxEpochs=numMaxEpochs, learningRate=learningRate, maxEpochsVal=maxEpochsVal);
            #=  Sin validación
            local ann;
            ann, = trainClassANN(topology,
                trainingInputs, trainingTargets,
                testInputs,     testTargets;
                maxEpochs=numMaxEpochs, learningRate=learningRate);
                =#
            (acc, _, _, _, _, _, F1, _) = confusionMatrix(collect(ann(testInputs')'), testTargets);

            testAccuraciesEachRepetition[numTraining] = acc;
            testF1EachRepetition[numTraining]         = F1;

        end;
        testAccuracies[numFold] = mean(testAccuraciesEachRepetition);
        testF1[numFold]         = mean(testF1EachRepetition);

    end;
    @show(topology)
    println("Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ", 100*std(testAccuracies));
    println("Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ", 100*std(testF1));
    println()
    return (topology, mean(testAccuracies), std(testAccuracies), mean(testF1), std(testF1))
end;

function rna_loop_1(inputs::Array{Float64,2}, targets::Array{Bool,2})
    nnpplayer = 18;
    for i in 1:nnpplayer
        (topology, meanTestAccuracies, stdTestAccuracies,
            meanTestF1, stdTestF1) = topology_test(inputs, targets, [i])
    end;
end;

function rna_loop_2(inputs::Array{Float64,2}, targets::Array{Bool,2})
    nnpplayer = 8;
    for i in 3:nnpplayer
        for j in 3:nnpplayer
            (topology, meanTestAccuracies, stdTestAccuracies,
                meanTestF1, stdTestF1) = topology_test(inputs, targets, [i,j])
        end;
    end;
    return (best_precision, best_f1, best_topology)
end;


seed!(1);

dataset_name="datasets/faces.data"
if (!isfile(dataset_name))
    (inputs, targets) = getInputs("datasets");
    println("Tamaños en la generación:")
    println(size(inputs))
    println(size(targets))
    write_dataset(dataset_name,inputs,targets)
end

# Cargamos el dataset
dataset = readdlm(dataset_name,',');

# Preparamos las entradas y las salidas deseadas
inputs = convert(Array{Float64,2}, dataset[:,1:6]);             #Array{Float64,2}
targets = oneHotEncoding(convert(Array{Any,1},dataset[:,7]));   #Array{Bool,2}


architecture_array  = [];#Array{String,1}(undef, 52);
acc_array  = [];# Array{Float64,1}(undef, 52);
f1_array  = [];# Array{Float64,1}(undef, 52);

@time begin
    (topology, testAccuracies, _, F1Score, _) = topology_test(inputs,targets,[0])
    push!(architecture_array, string(0));
    push!(acc_array, testAccuracies);
    push!(f1_array, F1Score);
    #rna_loop_1!(inputs,targets,architecture_array,acc_array,f1_array)
    #rna_loop_2!(inputs,targets,architecture_array,acc_array,f1_array)
    (topology, testAccuracies, _, F1Score, _) = topology_test(inputs,targets,[4,5,3])
    push!(architecture_array, string(4,",",5,",",3));
    push!(acc_array, testAccuracies);
    push!(f1_array, F1Score);
    #(topology, testAccuracies, _, F1Score, _) = topology_test(inputs,targets,[5,4,5])
    #push!(architecture_array, string(5,",",4,",",5));
    #push!(acc_array, testAccuracies);
    #push!(f1_array, F1Score);
    #(topology, testAccuracies, _, F1Score, _) = topology_test(inputs,targets,[4,8,3])
    #push!(architecture_array, string(4,",",8,",",3));
    #push!(acc_array, testAccuracies);
    #push!(f1_array, F1Score);
    #(topology, testAccuracies, _, F1Score, _) = topology_test(inputs,targets,[7,5,6])
    #push!(architecture_array, string(7,",",5,",",6));
    #push!(acc_array, testAccuracies);
    #push!(f1_array, F1Score);
    #(topology, testAccuracies, _, F1Score, _) = topology_test(inputs,targets,[5,5,7])
    #push!(architecture_array, string(5,",",5,",",7));
    #push!(acc_array, testAccuracies);
    #push!(f1_array, F1Score);
    #(topology, testAccuracies, _, F1Score, _) = topology_test(inputs,targets,[7,8,3])
    #push!(architecture_array, string(7,",",8,",",3));
    #push!(acc_array, testAccuracies);
    #push!(f1_array, F1Score);
    #(topology, testAccuracies, _, F1Score, _) = topology_test(inputs,targets,[5,6,4])
    #push!(architecture_array, string(5,",",6,",",4));
    #push!(acc_array, testAccuracies);
    #push!(f1_array, F1Score);
    #(topology, testAccuracies, _, F1Score, _) = topology_test(inputs,targets,[6,7,5])
    #push!(architecture_array, string(6,",",7,",",5));
    #push!(acc_array, testAccuracies);
    #push!(f1_array, F1Score);
    #(topology, testAccuracies, _, F1Score, _) = topology_test(inputs,targets,[4,5,4])
    #push!(architecture_array, string(4,",",5,",",4));
    #push!(acc_array, testAccuracies);
    #push!(f1_array, F1Score);
end;

graph=plot(architecture_array,[acc_array f1_array],title = "Precisión con diferentes topologías",label = ["Accurracy" "F1-score"],);
xlabel!("Topología");
ylabel!("Precisión");
display(plot(graph));