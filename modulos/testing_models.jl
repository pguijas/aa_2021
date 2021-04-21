include("datasets.jl")
include("attributes_from_dataset.jl")
include("models_cross_validation.jl")
include("graphics.jl")

# este archivo tiene funciones para probar los modelos, y devuelve gráficas con
# la std y media

function topology_test(inputs::Array{Float64,2}, targets::Array{Any,1}, topology::Array{Int64,1}, numFolds::Int64)

    learningRate = 0.01;
    numMaxEpochs = 1000;
    validationRatio = 0.2;
    maxEpochsVal = 30;
    numRepetitionsAANTraining = 50;

    classes = unique(targets);
    targets = oneHotEncoding(targets, classes);

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
    @show(topology);
    println("Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ", 100*std(testAccuracies));
    println("Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ", 100*std(testF1));
    println();
    return (topology, mean(testAccuracies), std(testAccuracies), mean(testF1), std(testF1));
end;

function rna_loop_1(inputs::Array{Float64,2}, targets::Array{Any,1}, nnpplayer::Int64, stlynn::Int64, numFolds::Int64)
    mean_acc = [];
    sdev_acc = [];
    mean_f1 = [];
    sdev_f1 = [];
    topologyarr = [];
    for i in stlynn:nnpplayer
        (topology, testAccuracies, testStd,
            testF1, F1Std) = topology_test(inputs, targets, [i], numFolds);
        push!(mean_acc,testAccuracies);
        push!(sdev_acc,testStd);
        push!(mean_f1,testF1);
        push!(sdev_f1,F1Std);
        push!(topologyarr, string("[",i,"]"));
    end;
    return (mean_acc, sdev_acc, mean_f1, sdev_f1, topologyarr);
end;

function rna_loop_2(inputs::Array{Float64,2}, targets::Array{Any,1}, nnpplayer::Int64, stlynn::Int64, numFolds::Int64)
    mean_acc = [];
    sdev_acc = [];
    mean_f1 = [];
    sdev_f1 = [];
    topologyarr = [];
    for i in stlynn:nnpplayer
        for j in stlynn:nnpplayer
            (topology, testAccuracies, testStd,
                testF1, F1Std) = topology_test(inputs, targets, [i], numFolds);
            push!(mean_acc,testAccuracies);
            push!(sdev_acc,testStd);
            push!(mean_f1,testF1);
            push!(sdev_f1,F1Std);
            push!(topologyarr, string("[",i,",",j,"]"));
        end;
    end;
    return (mean_acc, sdev_acc, mean_f1, sdev_f1, topologyarr);
end;

function testRNAs(inputs::Array{Float64,2}, targets::Array{Any,1}, parameters::Dict, numFolds::Int64, rep::Symbol)
    layers = parameters["layers"] = 1;
    if (layers==0)
        (_, testAccuracies, stdTestAccuracies, testF1, F1Std) = topology_test(inputs,targets,[0]);
        println("Test accuracies for a topology of 0 hidden layers: " + string(testAccuracies));
        println("Standard deviation for test accuracies for a topology of 0 hidden layers: " + string(testAccuracies));
    else
        if (layers==1)
            (mean_acc, sdev_acc, mean_f1, sdev_f1, topologyarr) = rna_loop_1(inputs, targets, parameters["maxNNxlayer"], parameters["fstNeuron"], numFolds);
        elseif (layers==2)
            (mean_acc, sdev_acc, mean_f1, sdev_f1, topologyarr) = rna_loop_2(inputs, targets, parameters["maxNNxlayer"], parameters["fstNeuron"], numFolds);
        else
            println("Test para arquitecturas de 3 capas o superiores no implementados.");
        end;
        printAccStdRNA(mean_acc, sdev_acc, topologyarr);
    end;
end;

function testSVM(inputs::Array{Float64,2}, targets::Array{Any,1}, parameters::Dict, numFolds::Int64, rep::Symbol)

    if (parameters["kernel"]=="linear")
        modelHyperparameters = Dict();
        modelHyperparameters["kernel"] = parameters["kernel"];
        modelHyperparameters["kernelDegree"] = 3;
        modelHyperparameters["kernelGamma"] = 2;
        modelHyperparameters["C"] = 1;
        (testAccuracies, testStd, testF1, F1Std) = modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, numFolds);
        println(string("Test accuracies for linear kernel: ",testAccuracies))
        println(string("Standard deviation for test accuracies of linear kernel: ",testStd))

    else
        mean_acc = [];
        sdev_acc = [];
        mean_f1 = [];
        sdev_f1 = [];
        if (parameters["kernel"]=="poly")
            for kernelGamma in 1:parameters["maxGamma"]
                modelHyperparameters = Dict();
                modelHyperparameters["kernel"] = parameters["kernel"];
                modelHyperparameters["kernelDegree"] = parameters["kernelDegree"];
                modelHyperparameters["kernelGamma"] = kernelGamma;
                modelHyperparameters["C"] = 1;
                (testAccuracies, testStd, _, _) = modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, numFolds);
                push!(mean_acc,testAccuracies);
                push!(sdev_acc,testStd);
                push!(mean_f1,testF1);
                push!(sdev_f1,F1Std);
            end;
        else
            for kernelGamma in 1:parameters["maxGamma"]
                modelHyperparameters = Dict();
                modelHyperparameters["kernel"] = "rbf";
                modelHyperparameters["kernelDegree"] = parameters["kernelDegree"];
                modelHyperparameters["kernelGamma"] = kernelGamma;
                modelHyperparameters["C"] = 1;
                (testAccuracies, testStd, _, _) = modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, numFolds);
                push!(mean_acc,testAccuracies);
                push!(sdev_acc,testStd);
                push!(mean_f1,testF1);
                push!(sdev_f1,F1Std);
            end;
        end;
        printAccStd(mean_acc, sdev_acc, parameters["maxGamma"], "Kernel Gamma");
    end;

end;

function testDecisionTree(inputs::Array{Float64,2}, targets::Array{Any,1}, maxDepth::Int64, numFolds::Int64, rep::Symbol)
    mean_acc = [];
    sdev_acc = [];
    mean_f1 = [];
    sdev_f1 = [];
    for depth in 1:maxDepth
        (testAccuracies, testStd, testF1, F1Std) = modelCrossValidation(:DecisionTree, Dict("maxDepth" => depth), inputs, targets, numFolds);
        push!(mean_acc,testAccuracies);
        push!(sdev_acc,testStd);
        push!(mean_f1,testF1);
        push!(sdev_f1,F1Std);
    end;
    printAccStd(mean_acc, sdev_acc, maxDepth, "maxDepth")
end;

function testKNN(inputs::Array{Float64,2}, targets::Array{Any,1}, max_Neigh::Int64, numFolds::Int64, rep::Symbol)
    mean_acc = [];
    sdev_acc = [];
    mean_f1 = [];
    sdev_f1 = [];
    for numNeighbors in 1:max_Neigh
        (testAccuracies, testStd, testF1, F1Std) = modelCrossValidation(:kNN, Dict("numNeighbors" => numNeighbors), inputs, targets, numFolds);
        push!(mean_acc,testAccuracies);
        push!(sdev_acc,testStd);
        push!(mean_f1,testF1);
        push!(sdev_f1,F1Std);
    end;
    printAccStd(mean_acc, sdev_acc, max_Neigh, "Number of Neighbors")
end;

function testingModels(modelType::Symbol, parameters::Dict, inputs::Array{Float64,2}, targets::Array{Any,1}, numFolds::Int64, rep::Symbol)
    if modelType==:ANN
        testRNAs(inputs,targets,parameters,numFolds,rep);
    elseif modelType==:SVM
        testSVM(inputs,targets,parameters,numFolds,rep);
    elseif modelType==:DecisionTree
        testDecisionTree(inputs,targets,parameters["maxDepth"],numFolds,rep)
    else
        testKNN(inputs,targets,parameters["maxNeighbors"],numFolds,rep)
    end;
end;
