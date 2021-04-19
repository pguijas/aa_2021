include("modulos/datasets.jl")
include("modulos/attributes_from_dataset.jl")
include("modulos/models_cross_validation.jl")
include("modulos/graphics.jl")


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
    @show(topology)
    println("Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ", 100*std(testAccuracies));
    println("Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ", 100*std(testF1));
    println()
    return (topology, mean(testAccuracies), std(testAccuracies), mean(testF1), std(testF1))
end;

function rna_loop_1(inputs::Array{Float64,2}, targets::Array{Any,1}, nnpplayer::Int64, stlynn::Int64, numFolds::Int64)
    mean_acc = [];
    sdev = [];
    topologyarr = [];
    for i in stlynn:nnpplayer
        (topology, meanTestAccuracies, stdTestAccuracies,
            _, _) = topology_test(inputs, targets, [i], numFolds)
        push!(mean_acc,testAccuracies);
        push!(sdev,testStd);
        push!(topologyarr, string("[",i,"]"));
    end;
    m = plot(topologyarr,mean_acc,title = "Accurracies",label = "Accurracy",);
    xlabel!("Topology");
    ylabel!("Precision");
    stdd = plot(topologyarr,sdev,title = "Standard Deviation",label = "std",);
    xlabel!("Topology");
    ylabel!("%");
    display(plot(m,stdd));
end;

function rna_loop_2(inputs::Array{Float64,2}, targets::Array{Any,1}, nnpplayer::Int64, stlynn::Int64, numFolds::Int64)
    mean_acc = [];
    sdev = [];
    topologyarr = [];
    for i in stlynn:nnpplayer
        for j in stlynn:nnpplayer
            (topology, meanTestAccuracies, stdTestAccuracies,
                _, _) = topology_test(inputs, targets, [i,j], numFolds)
            push!(mean_acc,testAccuracies);
            push!(sdev,testStd);
            push!(topologyarr, string("[",i,",",j,"]"));
        end;
    end;
    m = plot(topologyarr,mean_acc,title = "Accurracies",label = "Accurracy",);
    xlabel!("Topology");
    ylabel!("Precision");
    stdd = plot(topologyarr,sdev,title = "Standard Deviation",label = "std",);
    xlabel!("Topology");
    ylabel!("%");
    display(plot(m,stdd));
end;

function testRNAs(inputs::Array{Float64,2}, targets::Array{Any,1}, parameters::Dict, numFolds::Int64)
    layers = parameters["layers"] = 1;
    if (layers==0)
        (_, testAccuracies, stdTestAccuracies, _, _) = topology_test(inputs,targets,[0])
        println("Test accuracies for a topology of 0 hidden layers: " + string(testAccuracies))
        println("Standard deviation for test accuracies for a topology of 0 hidden layers: " + string(testAccuracies))
    elseif (layers==1)
        rna_loop_1(inputs, targets, parameters["maxNNxlayer"], parameters["fstNeuron"], numFolds)
    else (layers==2)
        rna_loop_2(inputs, targets, parameters["maxNNxlayer"], parameters["fstNeuron"], numFolds)
    end;
end;

function testSVM(inputs::Array{Float64,2}, targets::Array{Any,1}, parameters::Dict, numFolds::Int64)

    if (parameters["kernel"]=="linear")
        modelHyperparameters = Dict();
        modelHyperparameters["kernel"] = parameters["kernel"];
        modelHyperparameters["kernelDegree"] = 3;
        modelHyperparameters["kernelGamma"] = 2;
        modelHyperparameters["C"] = 1;
        (testAccuracies, testStd, _, _) = modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, numFolds);
        println("Test accuracies for a topology of 0 hidden layers: " + string(testAccuracies))
        println("Standard deviation for test accuracies for a topology of 0 hidden layers: " + string(testAccuracies))

    elseif (parameters["kernel"]=="poly")
        mean_acc = [];
        sdev = [];
        for kernelGamma in 1:parameters["maxGamma"]
            modelHyperparameters = Dict();
            modelHyperparameters["kernel"] = parameters["kernel"];
            modelHyperparameters["kernelDegree"] = parameters["kernelDegree"];
            modelHyperparameters["kernelGamma"] = kernelGamma;
            modelHyperparameters["C"] = 1;
            (testAccuracies, testStd, _, _) = modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, numFolds);
            push!(mean_acc,testAccuracies);
            push!(sdev,testStd);
            printAccStd(mean_acc, sdev, max_Neigh, "Kernel Gamma")
        end;

    else
        mean_acc = [];
        sdev = [];
        for kernelGamma in 1:parameters["maxGamma"]
            modelHyperparameters = Dict();
            modelHyperparameters["kernel"] = "rbf";
            modelHyperparameters["kernelDegree"] = parameters["kernelDegree"];
            modelHyperparameters["kernelGamma"] = kernelGamma;
            modelHyperparameters["C"] = 1;
            (testAccuracies, testStd, _, _) = modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, numFolds);
            push!(mean_acc,testAccuracies);
            push!(sdev,testStd);
        end;
        printAccStd(mean_acc, sdev, max_Neigh, "Kernel Gamma")
    end;

end

function testDecisionTree(inputs::Array{Float64,2}, targets::Array{Any,1}, maxDepth::Int64, numFolds::Int64)
    mean_acc = [];
    sdev = [];
    for depth in 1:maxDepth
        (testAccuracies, testStd, _, _) = modelCrossValidation(:DecisionTree, Dict("maxDepth" => depth), inputs, targets, numFolds);
        push!(mean_acc,testAccuracies);
        push!(sdev,testStd);
    end;
    printAccStd(mean_acc, sdev, maxDepth, "maxDepth")
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
        testRNAs(inputs,targets,parameters,numFolds);
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

# Entrenamos las RR.NN.AA.
modelHyperparameters = Dict();
modelHyperparameters["fstNeuron"] = 1;
modelHyperparameters["maxNNxlayer"] = 2;
modelHyperparameters["layers"] = 1;
testingModels(:ANN, modelHyperparameters, inputs, targets, numFolds);

# Entrenamos las SVM
#modelHyperparameters = Dict();
#testingModels(:SVM, modelHyperparameters, inputs, targets, numFolds);

# Entrenamos los arboles de decision
testingModels(:KNN, Dict("maxNeighbors" => 20), inputs, targets, numFolds);

# Entrenamos los arboles de decision
testingModels(:DecisionTree, Dict("maxDepth" => 20), inputs, targets, numFolds);

modelHyperparameters = Dict();
modelHyperparameters["kernel"] = 1;
modelHyperparameters["kernelDegree"] = 2;
modelHyperparameters["maxGamma"] = 1;
testingModels(:SVM, modelHyperparameters, inputs, targets, numFolds);
