
#Confusion 1 clase
function confusionMatrix_P(outputs::Array{Bool,1}, targets::Array{Bool,1})
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
function confusionMatrix_P(outputs::Array{Bool,2}, targets::Array{Bool,2},dataInRows::Bool=true)
    @assert(all(size(outputs).==size(targets)));
    
    #Cálculo matriz
    
    #Matriz de confusión: filas(reales) columnas(predicciones)
    n_patrones= dataInRows ? size(outputs,1) : size(outputs,2)
    n_clases= dataInRows ? size(outputs,2) : size(outputs,1)
    #comprobar si realmente es de 1d
    if (n_clases==1)
        if (dataInRows)
            return confusionMatrix_P(outputs[:,1],targets[:,1])
        else
            return confusionMatrix_P(outputs[1,:],targets[1,:])
        end
    else
        matriz=convert(Array{Int},zeros(n_clases,n_clases))
        for i in 1:n_patrones
            #buscar alguna func del palo de dame el indice del elemento que sea true
            if (dataInRows)

                real=primero_que_cumple(outputs[i,:])
                prediccion=primero_que_cumple(targets[i,:])
            else

                real=primero_que_cumple(outputs[:,i])
                prediccion=primero_que_cumple(targets[:,i])
            end
            matriz[real,prediccion]=matriz[real,prediccion]+1
        end
    end
    return matriz
end


confusionMatrix_P(outputs::Array{Float64,1}, targets::Array{Bool,1}; threshold::Float64=0.5) = confusionMatrix_P(Array{Bool,1}(outputs.>=threshold), targets);
#Pasamos de Float32 a Float64
confusionMatrix_P(outputs::Array{Float32,1}, targets::Array{Bool,1}; threshold::Float64=0.5) = confusionMatrix_P(Float64.(outputs), targets; threshold=threshold);
confusionMatrix_P(outputs::Array{Float32,2}, targets::Array{Bool,2}; dataInRows::Bool=true)  = confusionMatrix_P(Float64.(outputs), targets; dataInRows=dataInRows);

