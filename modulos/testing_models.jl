include("datasets.jl")
include("attributes_from_dataset.jl")
include("models_cross_validation.jl")
include("graphics.jl")

# este archivo tiene funciones para probar los modelos, y devuelve gráficas con
# la std y media

function topology_test(inputs::Array{Float64,2}, targets::Array{Any,1}, topology::Array{Int64,1}, numFolds::Int64)

    @show(topology);
    learningRate = 0.01;
    numMaxEpochs = 1000;
    validationRatio = 0.2;
    maxEpochsVal = 30;
    numRepetitionsAANTraining = 50;

    classes = unique(targets);
    targets = oneHotEncoding(targets, classes);

    μ_testAcc = Array{Float64,1}(undef, numFolds);
    μ_testF1         = Array{Float64,1}(undef, numFolds);

    crossValidationIndices = crossvalidation(size(inputs,1), numFolds);

    for numFold in 1:numFolds

        local trainingInputs, testInputs, trainingTargets, testTargets;
        trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
        testInputs        = inputs[crossValidationIndices.==numFold,:];
        trainingTargets   = targets[crossValidationIndices.!=numFold,:];
        testTargets       = targets[crossValidationIndices.==numFold,:];

        μ_testAccEachRepetition = Array{Float64,1}(undef, numRepetitionsAANTraining);
        μ_testF1EachRepetition         = Array{Float64,1}(undef, numRepetitionsAANTraining);

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

            μ_testAccEachRepetition[numTraining] = acc;
            μ_testF1EachRepetition[numTraining]         = F1;

        end;
        μ_testAcc[numFold] = mean(μ_testAccEachRepetition);
        μ_testF1[numFold]         = mean(μ_testF1EachRepetition);

    end;
    println("Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(μ_testAcc), ", with a standard deviation of ", 100*std(μ_testAcc));
    println("Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100*mean(μ_testF1), ", with a standard deviation of ", 100*std(μ_testF1));
    println();
    return (topology, mean(μ_testAcc), std(μ_testAcc), mean(μ_testF1), std(μ_testF1));
end;

function rna_loop_1(inputs::Array{Float64,2}, targets::Array{Any,1}, nnpplayer::Int64, stlynn::Int64, numFolds::Int64)
    μ_acc = [];
    σ_acc = [];
    μ_f1 = [];
    σ_f1 = [];
    topologyarr = [];
    for i in stlynn:nnpplayer
        (topology, μ_testAcc, σ_stdAcc,
            μ_testF1, σ_stdF1) = topology_test(inputs, targets, [i], numFolds);
        push!(μ_acc,μ_testAcc);
        push!(σ_acc,σ_stdAcc);
        push!(μ_f1,μ_testF1);
        push!(σ_f1,σ_stdF1);
        push!(topologyarr, string("[",i,"]"));
    end;
    return (μ_acc, σ_acc, μ_f1, σ_f1, topologyarr);
end;

function rna_loop_2(inputs::Array{Float64,2}, targets::Array{Any,1}, nnpplayer::Int64, stlynn::Int64, numFolds::Int64)
    μ_acc = [];
    σ_acc = [];
    μ_f1 = [];
    σ_f1 = [];
    topologyarr = [];
    for i in stlynn:nnpplayer
        for j in stlynn:nnpplayer
            (topology, μ_testAcc, σ_stdAcc,
                μ_testF1, σ_stdF1) = topology_test(inputs, targets, [i,j], numFolds);
            push!(μ_acc,μ_testAcc);
            push!(σ_acc,σ_stdAcc);
            push!(μ_f1,μ_testF1);
            push!(σ_f1,σ_stdF1);
            push!(topologyarr, string("[",i,",",j,"]"));
        end;
    end;
    return (μ_acc, σ_acc, μ_f1, σ_f1, topologyarr);
end;

function testRNAs(inputs::Array{Float64,2}, targets::Array{Any,1}, parameters::Dict, numFolds::Int64, rep::Symbol)
    layers = parameters["layers"];
    if (layers==0)
        (_, μ_testAcc, stdμ_testAcc, μ_testF1, σ_stdF1) = topology_test(inputs,targets,[0]);
        println("Test accuracies for a topology of 0 hidden layers: " + string(μ_testAcc));
        println("Standard deviation for test accuracies for a topology of 0 hidden layers: " + string(μ_testAcc));
    else
        if (layers==1)
            (μ_acc, σ_acc, μ_f1, σ_f1, topologyarr) = rna_loop_1(inputs, targets, parameters["maxNNxlayer"], parameters["fstNeuron"], numFolds);
        elseif (layers==2)
            (μ_acc, σ_acc, μ_f1, σ_f1, topologyarr) = rna_loop_2(inputs, targets, parameters["maxNNxlayer"], parameters["fstNeuron"], numFolds);
        else
            println("Test para arquitecturas de 3 capas o superiores no implementados.");
        end;
        printAccStdRNA(μ_acc, σ_acc, μ_f1, σ_f1, topologyarr, rep);
    end;
end;

function testSVM(inputs::Array{Float64,2}, targets::Array{Any,1}, parameters::Dict, numFolds::Int64, rep::Symbol)
    kernel = parameters["kernel"];
    if (kernel=="linear")
        modelHyperparameters = Dict();
        modelHyperparameters["kernel"] = kernel;
        modelHyperparameters["kernelDegree"] = 3;
        modelHyperparameters["kernelGamma"] = 2;
        modelHyperparameters["C"] = 1000000;
        (μ_testAcc, σ_stdAcc, μ_testF1, σ_stdF1) = modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, numFolds);
        println(string("Test accuracies for linear kernel: ",μ_testAcc))
        println(string("Standard deviation for test accuracies of linear kernel: ",σ_stdAcc))

    else
        μ_acc = [];
        σ_acc = [];
        μ_f1 = [];
        σ_f1 = [];
        kernel° = parameters["kernelDegree"];
        max_γ = parameters["maxGamma"];
        plot3d = rep==:Plot3D;
        if (kernel=="poly")
            for degree in 1:kernel°
                if !plot3d
                    local_μ_acc = []
                    local_σ_acc = []
                    local_μ_f1 = []
                    local_σ_f1 = []
                end;
                for γ in 1:max_γ
                    modelHyperparameters = Dict();
                    modelHyperparameters["kernel"] = kernel;
                    modelHyperparameters["kernelDegree"] = degree;
                    modelHyperparameters["kernelGamma"] = γ;
                    modelHyperparameters["C"] = 1000000;
                    (μ_testAcc, σ_stdAcc, μ_testF1, σ_stdF1) = modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, numFolds);
                    if plot3d
                        push!(μ_acc,μ_testAcc);
                        push!(σ_acc,σ_stdAcc);
                        push!(μ_f1,μ_testF1);
                        push!(σ_f1,σ_stdF1);
                    else
                        push!(local_μ_acc,μ_testAcc);
                        push!(local_σ_acc,σ_stdAcc);
                        push!(local_μ_f1,μ_testF1);
                        push!(local_σ_f1,σ_stdF1);
                    end;
                end;
                if !plot3d
                    printAccStd(local_μ_acc, local_σ_acc, local_μ_f1, local_σ_f1, max_γ, "Kernel γ", rep);
                end;
            end;
            if plot3d
                pyplot();
                plot(1:kernel°, 1:max_γ, μ_acc, st=:surface, xlabel = "Kernel°", ylabel = "Kernel γ", camera=(22,30));
            end;

        else
            for γ in 1:max_γ
                modelHyperparameters = Dict();
                modelHyperparameters["kernel"] = "rbf";
                modelHyperparameters["kernelDegree"] = kernel°;
                modelHyperparameters["kernelGamma"] = γ;
                modelHyperparameters["C"] = 1000000;
                (μ_testAcc, σ_stdAcc, μ_testF1, σ_stdF1) = modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, numFolds);
                push!(μ_acc,μ_testAcc);
                push!(σ_acc,σ_stdAcc);
                push!(μ_f1,μ_testF1);
                push!(σ_f1,σ_stdF1);
            end;
            printAccStd(μ_acc, σ_acc, μ_f1, σ_f1, max_γ, "Kernel γ", rep);
        end;
    end;

end;

function testDecisionTree(inputs::Array{Float64,2}, targets::Array{Any,1}, maxDepth::Int64, numFolds::Int64, rep::Symbol)
    μ_acc = [];
    σ_acc = [];
    μ_f1 = [];
    σ_f1 = [];
    for depth in 1:maxDepth
        (μ_testAcc, σ_stdAcc, μ_testF1, σ_stdF1) = modelCrossValidation(:DecisionTree, Dict("maxDepth" => depth), inputs, targets, numFolds);
        push!(μ_acc,μ_testAcc);
        push!(σ_acc,σ_stdAcc);
        push!(μ_f1,μ_testF1);
        push!(σ_f1,σ_stdF1);
    end;
    printAccStd(μ_acc, σ_acc, μ_f1, σ_f1, maxDepth, "maxDepth", rep);
end;

function testKNN(inputs::Array{Float64,2}, targets::Array{Any,1}, max_Neigh::Int64, numFolds::Int64, rep::Symbol)
    μ_acc = [];
    σ_acc = [];
    μ_f1 = [];
    σ_f1 = [];
    for numNeighbors in 1:max_Neigh
        (μ_testAcc, σ_stdAcc, μ_testF1, σ_stdF1) = modelCrossValidation(:kNN, Dict("numNeighbors" => numNeighbors), inputs, targets, numFolds);
        push!(μ_acc,μ_testAcc);
        push!(σ_acc,σ_stdAcc);
        push!(μ_f1,μ_testF1);
        push!(σ_f1,σ_stdF1);
    end;
    printAccStd(μ_acc, σ_acc, μ_f1, σ_f1, max_Neigh, "Number of Neighbors", rep);
end;

function testingModels(modelType::Symbol, parameters::Dict, inputs::Array{Float64,2}, targets::Array{Any,1}, numFolds::Int64; rep::Symbol=:All)
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
