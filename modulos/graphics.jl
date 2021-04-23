# =============================================================================
# graphics.jl -> Funciones útiles para visualizar gráficamente los resultados de una RNA:
#   -
# =============================================================================

using Plots;

function print_train_results(
    trainingLosses::Array{Float64,1},validationLosses::Array{Float64,1},testLosses::Array{Float64,1},
    trainingAccuracies::Array{Float64,1},validationAccuracies::Array{Float64,1},testAccuracies::Array{Float64,1})
    x=1:size(trainingLosses,1)
    losses=plot(x,[trainingLosses validationLosses testLosses],title = "Loss",label = ["Training" "Validation" "Test"],);
    xlabel!("Epoch");
    ylabel!("Error");
    accuracy=plot(x,[trainingAccuracies validationAccuracies testAccuracies],title = "Accuracies",label = ["Training" "Validation" "Test"],);
    xlabel!("Epoch");
    ylabel!("% Accuracies");
    display(plot(losses,accuracy));
end;


function printAccStd(mean_acc::Array{Any,1}, sdev::Array{Any,1}, mean_f1::Array{Any,1}, sdev_f1::Array{Any,1}, N::Int64, xlabel::String, rep::Symbol)
    if rep==:All
        m = plot([1:N],[mean_acc mean_f1],title = "Mean Values",label = ["Accurracies" "F1-Score"]);
        xlabel!(xlabel);
        ylabel!("Precision");
        stdd = plot([1:N],[sdev sdev_f1],title = "Standard Deviation",label = ["Accurracies" "F1-Score"]);
        xlabel!(xlabel);
        #ylabel!("%");
        display(plot(m,stdd));
    elseif rep==:AccStd
        m = plot([1:N],mean_acc,title = "Mean Values",label = "Accurracies");
        xlabel!(xlabel);
        ylabel!("Precision");
        stdd = plot([1:N],sdev,title = "Standard Deviation",label = "Standard Deviation");
        xlabel!(xlabel);
        #ylabel!("%");
        display(plot(m,stdd));
    elseif rep==:F1Std
        m = plot([1:N],mean_f1,title = "Mean Values",label = "F1-Score");
        xlabel!(xlabel);
        ylabel!("Precision");
        stdd = plot([1:N],sdev_f1,title = "Standard Deviation",label = "Standard Deviation");
        xlabel!(xlabel);
        #ylabel!("%");
        display(plot(m,stdd));
    elseif rep==:AccF1
        m = plot([1:N],[mean_acc mean_f1],title = "Mean Values",label = ["Accurracies" "F1-Score"]);
        xlabel!(xlabel);
        ylabel!("Precision");
        display(m);
    end;
end;

function printAccStdRNA(mean_acc::Array{Any,1}, sdev::Array{Any,1}, mean_f1::Array{Any,1}, sdev_f1::Array{Any,1}, topologyarr::Array{Any,1}, rep::Symbol)
    m = plot(topologyarr,mean_acc,title = "Accurracies",label = "Accurracy",);
    xlabel!("Topology");
    ylabel!("Precision");
    stdd = plot(topologyarr,sdev,title = "Standard Deviation",label = "std",);
    xlabel!("Topology");
    #ylabel!("%");
    display(plot(m,stdd));
end;
