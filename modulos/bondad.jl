# =============================================================================
# bondad.jl -> Funciones útiles calcular la bondad de un sistema:
#   - accuracy (precisión)
#   -> falta p4
# =============================================================================

#Precisión bool 1d
accuracy(outputs::Array{Bool,1}, targets::Array{Bool,1}) = mean(outputs.==targets);

#Precisión bool 2d
function accuracy(outputs::Array{Bool,2}, targets::Array{Bool,2}; dataInRows::Bool=true)
    @assert(all(size(outputs).==size(targets)));
    if (dataInRows)
        # Cada patron esta en cada fila
        if (size(targets,2)==1)
            return accuracy(outputs[:,1], targets[:,1]);
        else
            classComparison = targets .== outputs
            correctClassifications = all(classComparison, dims=2)
            return mean(correctClassifications)
        end;
    else
        # Cada patron esta en cada columna
        if (size(targets,1)==1)
            return accuracy(outputs[1,:], targets[1,:]);
        else
            classComparison = targets .== outputs
            correctClassifications = all(classComparison, dims=1)
            return mean(correctClassifications)
        end;
    end;
end;

#Precisión Float64 1d
accuracy(outputs::Array{Float64,1}, targets::Array{Bool,1}; threshold::Float64=0.5) = accuracy(Array{Bool,1}(outputs.>=threshold), targets);

#Precisión Float64 2d
function accuracy(outputs::Array{Float64,2}, targets::Array{Bool,2}; dataInRows::Bool=true)
    @assert(all(size(outputs).==size(targets)));
    if (dataInRows)
        # Cada patron esta en cada fila
        if (size(targets,2)==1)
            return accuracy(outputs[:,1], targets[:,1]);
        else
            vmax = maximum(outputs, dims=2);
            outputs = Array{Bool,2}(outputs .== vmax);
            return accuracy(outputs, targets);
        end;
    else
        # Cada patron esta en cada columna
        if (size(targets,1)==1)
            return accuracy(outputs[1,:], targets[1,:]);
        else
            vmax = maximum(outputs, dims=1)
            outputs = Array{Bool,2}(outputs .== vmax)
            return accuracy(outputs, targets; dataInRows=false);
        end;
    end;
end;

# Añado estas funciones porque las RR.NN.AA. dan la salida como matrices de valores Float32 en lugar de Float64
# Con estas funciones se pueden usar indistintamente matrices de Float32 o Float64
accuracy(outputs::Array{Float32,1}, targets::Array{Bool,1}; threshold::Float64=0.5) = accuracy(Float64.(outputs), targets; threshold=threshold);
accuracy(outputs::Array{Float32,2}, targets::Array{Bool,2}; dataInRows::Bool=true)  = accuracy(Float64.(outputs), targets; dataInRows=dataInRows);


#Confusion 1 clase
function confusionMatrix(outputs::Array{Bool,1}, targets::Array{Bool,1})
    @assert(all(size(outputs).==size(targets)));
    #Matriz de confusión: filas(reales) columnas(predicciones)
    matriz=convert(Array{Int},zeros(2,2))
    for i in 1:size(outputs,1)
        if (targets[i])
            real=2
        else
            real=1
        end
        if (outputs[i])
            prediccion=2
        else
            prediccion=1
        end
        matriz[real,prediccion]=matriz[real,prediccion]+1
    end
    return matriz
end

function primero_que_cumple(array::Array{Bool,1})
    for i in 1:size(array,1)
        if (array[i])
            return i
        end
    end
end

#Confusion +1 clase, de momento trabajo con patrones x columnas
function confusionMatrix(outputs::Array{Bool,2}, targets::Array{Bool,2},dataInRows::Bool=true)
    @assert(all(size(outputs).==size(targets)));
    #Matriz de confusión: filas(reales) columnas(predicciones)
    #comprobar si realmente es de 1d

    n_patrones= dataInRows ? size(outputs,1) : size(outputs,2)
    n_clases= dataInRows ? size(outputs,2) : size(outputs,1)
    matriz=convert(Array{Int},zeros(n_clases,n_clases))
    for i in 1:n_patrones
        println(i)
        #buscar alguna func del palo de dame el indice del elemento que sea true
        if (dataInRows)

            real=primero_que_cumple(targets[i,:])
            prediccion=primero_que_cumple(targets[i,:])
        else

            real=primero_que_cumple(targets[:,i])
            prediccion=primero_que_cumple(targets[:,i])
        end
        matriz[real,prediccion]=matriz[real,prediccion]+1
    end
    return matriz
end
