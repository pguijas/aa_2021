include("modulos/datasets.jl")
include("modulos/attributes_from_dataset.jl")
include("modulos/models_cross_validation.jl")
include("modulos/graphics.jl")


function topology_test(inputs::Array{Float64,2}, targets::Array{Bool,2}, topology::Array{Int64,1}, numFolds::Int64)

    learningRate = 0.01;
    numMaxEpochs = 1000;
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

function rna_loop_1(inputs::Array{Float64,2}, targets::Array{Bool,2}, nnpplayer::Int64, stlynn::Int64, numFolds::Int64)
    for i in stlynn:nnpplayer
        (topology, meanTestAccuracies, stdTestAccuracies,
            meanTestF1, stdTestF1) = topology_test(inputs, targets, [i], numFolds)
    end;
end;

function rna_loop_2(inputs::Array{Float64,2}, targets::Array{Bool,2}, nnpplayer::Int64, stlynn::Int64, numFolds::Int64)
    for i in stlynn:nnpplayer
        for j in stlynn:nnpplayer
            (topology, meanTestAccuracies, stdTestAccuracies,
                meanTestF1, stdTestF1) = topology_test(inputs, targets, [i,j], numFolds)
        end;
    end;
end;

function testDecisionTree(inputs::Array{Float64,2}, targets::Array{Any,1}, maxDepth::Int64, numFolds::Int64)
    mean_acc = [];
    sdev = [];

    for depth in 1:maxDepth
        (testAccuracies, testStd, _, _) = modelCrossValidation(:DecisionTree, Dict("maxDepth" => depth), inputs, targets, numFolds);
        push!(mean_acc,testAccuracies);
        push!(sdev,testStd);
    end;

    m = plot([1:maxDepth],mean_acc,title = "Accurracies",label = "Accurracy",);
    xlabel!("maxDepth");
    ylabel!("Precision");
    stdd = plot([1:maxDepth],sdev,title = "Standard Deviation",label = "std",);
    xlabel!("maxDepth");
    ylabel!("%");
    display(plot(m,stdd))
end;

function testKNN(inputs::Array{Float64,2}, targets::Array{Any,1}, max_Neigh::Int64, numFolds::Int64)
    mean_acc = [];
    sdev = [];

    for numNeighbors in 1:max_Neigh
        (testAccuracies, testStd, _, _) = modelCrossValidation(:kNN, Dict("numNeighbors" => numNeighbors), inputs, targets, numFolds);
        push!(mean_acc,testAccuracies);
        push!(sdev,testStd);
    end;

    printAccStd(mean_acc, sdev, max_Neigh, "Number of Neighbors")
end

function testingModels(modelType::Symbol, parameters::Dict, inputs::Array{Float64,2}, targets::Array{Any,1}, numFolds::Int64)#, topology::Array{Int64,1})
    if modelType==:ANN
        rna_loop_1(inputs,targets,parameters["maxNNxlayer"],parameters["fstNeuron"],numFolds);
    elseif modelType==:SVM
        println("svm")
    elseif modelType==:DecisionTree
        testDecisionTree(inputs,targets,parameters["maxDepth"],numFolds)
    else
        testKNN(inputs,targets,parameters["maxNeighbors"],numFolds)
    end;
end


dataset_name="datasets/faces.data"
if (!isfile(dataset_name))
    (inputs, targets) = getInputs("datasets");
    println("Tamaños en la generación:")
    println(size(inputs))
    println(size(targets))
    write_dataset(dataset_name,inputs,targets)
end

dataset = readdlm(dataset_name,',');

inputs = convert(Array{Float64,2}, dataset[:,1:6]);
targets = convert(Array{Any,1},dataset[:,7]);

seed!(1);

numFolds = 10;
#==
# Entrenamos las RR.NN.AA.
modelHyperparameters = Dict();
modelHyperparameters["fstNeuron"] = 1;
modelHyperparameters["maxNNxlayer"] = 10;
testingModels(:ANN, modelHyperparameters, inputs, targets, numFolds);

# Entrenamos las SVM
modelHyperparameters = Dict();
testingModels(:SVM, modelHyperparameters, inputs, targets, numFolds);
=#
# Entrenamos los arboles de decision
testingModels(:KNN, Dict("maxNeighbors" => 20), inputs, targets, numFolds);

# Entrenamos los kNN
#testingModels(:kNN, Dict(), inputs, targets, numFolds);
