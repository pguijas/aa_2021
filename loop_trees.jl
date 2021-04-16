include("modulos/datasets.jl")
include("modulos/attributes_from_dataset.jl")
include("modulos/models_cross_validation.jl")
using Plots;
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
maxDepth = 30;
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
display(plot(m,stdd));
