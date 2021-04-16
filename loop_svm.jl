include("modulos/datasets.jl")
include("modulos/attributes_from_dataset.jl")
include("modulos/models_cross_validation.jl")

seed!(1);

dataset_name="datasets/faces.data"
if (!isfile(dataset_name))
    (inputs, targets) = getInputs("../AA_DATASET");
    println("Tamaños en la generación:")
    println(size(inputs))
    println(size(targets))
    write_dataset(dataset_name,inputs,targets)
end

dataset = readdlm(dataset_name,',');
inputs = convert(Array{Float64,2}, dataset[:,1:6]);             #Array{Float64,2}
targets = convert(Array{Any,1},dataset[:,7]);                   #Array{Bool,2}


numFolds = 10;

# Parametros del SVM
kernel = "linear"; #linear, poly, rbf, sigmoid, precomputed
kernelDegree = 3; # en caso de que sea 'poly', indica el grado del polinomio (num imputs)
kernelGamma = 2; #coeficiente del kernel para rbf, poly y sigmoid
C=1; #parámetro de regularización

modelHyperparameters = Dict();
modelHyperparameters["kernel"] = kernel;
modelHyperparameters["kernelDegree"] = kernelDegree;
modelHyperparameters["kernelGamma"] = kernelGamma;
modelHyperparameters["C"] = C;
modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, numFolds);

max_σ = 25:
mean_acc = [];
sdev = [];
for kernelGamma in 2:max_σ
    kernel = "rbf";
    modelHyperparameters = Dict();
    modelHyperparameters["kernel"] = kernel;
    modelHyperparameters["kernelDegree"] = kernelDegree;
    modelHyperparameters["kernelGamma"] = kernelGamma;
    modelHyperparameters["C"] = C;
    (testAccuracies, testStd, _, _) = modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, numFolds);
    push!(mean_acc,testAccuracies);
    push!(sdev,testStd);
end;

m = plot([2:max_σ],mean_acc,title = "Accurracies",label = "Accurracy",);
xlabel!("σ");
ylabel!("Precision");
stdd = plot([2:max_σ],sdev,title = "Standard Deviation",label = "std",);
xlabel!("σ");
ylabel!("%");
display(plot(m,stdd));


mean_acc = [];
sdev = [];
kernelDegree = 5;
for kernelGamma in 2:max_σ
    kernel = "poly";
    modelHyperparameters = Dict();
    modelHyperparameters["kernel"] = kernel;
    modelHyperparameters["kernelDegree"] = kernelDegree;
    modelHyperparameters["kernelGamma"] = kernelGamma;
    modelHyperparameters["C"] = C;
    (testAccuracies, testStd, _, _) = modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, numFolds);
    push!(mean_acc,testAccuracies);
    push!(sdev,testStd);
end;

m = plot([2:max_σ],mean_acc,title = "Accurracies",label = "Accurracy",);
xlabel!("σ");
ylabel!("Precision");
stdd = plot([2:max_σ],sdev,title = "Standard Deviation",label = "std",);
xlabel!("σ");
ylabel!("%");
display(plot(m,stdd));
