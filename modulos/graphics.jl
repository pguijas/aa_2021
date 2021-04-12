# =============================================================================
# graphics.jl -> Funciones útiles para visualizar gráficamente los resultados de una RNA:
#   -
# =============================================================================

using Plots

function print_train_results(
    trainingLosses::Array{Float64,1},validationLosses::Array{Float64,1},testLosses::Array{Float64,1},
    trainingAccuracies::Array{Float64,1},validationAccuracies::Array{Float64,1},testAccuracies::Array{Float64,1})
    x=1:size(trainingLosses,1)
    losses=plot(x,[trainingLosses validationLosses testLosses],title = "Loss",label = ["Training" "Validation" "Test"],);
    xlabel!("Epoch");
    ylabel!("Error)");
    accuracy=plot(x,[trainingAccuracies validationAccuracies testAccuracies],title = "Accuracies",label = ["Training" "Validation" "Test"],);
    xlabel!("Epoch");
    ylabel!("% Accuracies");
    display(plot(losses,accuracy));
end

# en la cpu trabajamos con float 32
function print_train_results(
    trainingLosses::Array{Float32,1},validationLosses::Array{Float32,1},testLosses::Array{Float32,1},
    trainingAccuracies::Array{Float32,1},validationAccuracies::Array{Float32,1},testAccuracies::Array{Float32,1})
    x=1:size(trainingLosses,1)
    losses=plot(x,[trainingLosses validationLosses testLosses],title = "Loss",label = ["Training" "Validation" "Test"],);
    xlabel!("Epoch");
    ylabel!("Error)");
    accuracy=plot(x,[trainingAccuracies validationAccuracies testAccuracies],title = "Accuracies",label = ["Training" "Validation" "Test"],);
    xlabel!("Epoch");
    ylabel!("% Accuracies");
    display(plot(losses,accuracy));
end
