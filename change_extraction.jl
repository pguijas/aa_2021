include("modulos/datasets.jl")
include("modulos/attributes_from_dataset.jl")

function createDataset(dataset_name::String, extraction::Symbol)
    (inputs, targets) = getInputs("datasets"; extr=extraction);
    println("Tamaños en la generación:");
    println(size(inputs));
    println(size(targets));
    write_dataset(dataset_name,inputs,targets);
    while (!isfile("datasets/faces.data"))
        sleep(1);
    end;
end;

#=
hay tres extracciones
    · :A1 (aproximación 1)
    · :A21 (aproximación 2 extracción 1)
    · :A22 (aproximación 2 extracción 2)
=#
extraction = :A22;
dataset_name="datasets/faces.data";
createDataset(dataset_name,extraction);

if extraction==:A1
    x = 6;
    y = 7;
elseif extraction==:A21
    x = 42;
    y = 43;
elseif extraction==:A22
    x = 36;
    y = 37;
end;

dataset = readdlm(dataset_name,',');
inputs = convert(Array{Float64,2}, dataset[:,1:x]);
targets = convert(Array{Any,1},dataset[:,y]);
