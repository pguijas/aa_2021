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
max_Neigh = 30;
mean_acc = [];
sdev = [];

for numNeighbors in 1:max_Neigh
    (testAccuracies, testStd, _, _) = modelCrossValidation(:kNN, Dict("numNeighbors" => numNeighbors), inputs, targets, numFolds);
    push!(mean_acc,testAccuracies);
    push!(sdev,testStd);
end;

m = plot([1:max_Neigh],mean_acc,title = "Accurracies",label = "Accurracy",);
xlabel!("σ");
ylabel!("Precision");
stdd = plot([1:max_Neigh],sdev,title = "Standard Deviation",label = "std",);
xlabel!("σ");
ylabel!("%");
display(plot(m,stdd));
