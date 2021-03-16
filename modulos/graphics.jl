# =============================================================================
# graphics.jl -> Funciones útiles para visualizar gráficamente los resultados de una RNA:
#   -
# =============================================================================

using Plots;
using PyPlot;

function print_train_results(
    trainingLosses::Array{Float64,1},validationLosses::Array{Float64,1},testLosses::Array{Float64,1},
    trainingAccuracies::Array{Float64,1},validationAccuracies::Array{Float64,1},testAccuracies::Array{Float64,1})
    x=1:size(trainingLosses,1)
    losses=plot(x,[trainingLosses validationLosses testLosses],title = "Loss",label = ["Training" "Validation" "Test"],)
    xlabel!("Epoch")
    ylabel!("Error (CrossEntropy (Creo))")
    accuracy=plot(x,[trainingAccuracies validationAccuracies testAccuracies],title = "Accuracies",label = ["Training" "Validation" "Test"],)
    xlabel!("Epoch")
    ylabel!("% Accuracies")
    plot(losses,accuracy)
end
