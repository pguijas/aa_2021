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
    n_patrones= dataInRows ? size(outputs,1) : size(outputs,2)
    n_clases= dataInRows ? size(outputs,2) : size(outputs,1)
    #comprobar si realmente es de 1d
    if (n_clases==1)
        if (dataInRows)
            return confusionMatrix(outputs[:,1],targets[:,1])
        else
            return confusionMatrix(outputs[1,:],targets[1,:])
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
        return matriz
    end
end

#Esto sería interesante meterlo en otro lado
float_outputs_to_bool(outputs::Array{Float64,1},threshold::Float64=0.5)=Array{Bool,1}(outputs.>=threshold)
function float_outputs_to_bool(outputs::Array{Float64,2},dataInRows::Bool,threshold::Float64=0.5)
    if (dataInRows)
        # Cada patron esta en cada fila
        if (size(targets,2)==1)
            return float_outputs_to_bool(outputs[:,1],threshold);
        else
            vmax = maximum(outputs, dims=2);
            return Array{Bool,2}(outputs .== vmax);
        end;
    else
        # Cada patron esta en cada columna
        if (size(targets,1)==1)
            return float_outputs_to_bool(outputs[1,:],threshold);
        else
            vmax = maximum(outputs, dims=1)
            return Array{Bool,2}(outputs .== vmax)
        end;
    end;
end;

confusionMatrix(outputs::Array{Float64,1}, targets::Array{Bool,1}; threshold::Float64=0.5) = confusionMatrix(Array{Bool,1}(outputs.>=threshold), targets);
#Pasamos de Float32 a Float64
confusionMatrix(outputs::Array{Float32,1}, targets::Array{Bool,1}; threshold::Float64=0.5) = confusionMatrix(Float64.(outputs), targets; threshold=threshold);
confusionMatrix(outputs::Array{Float32,2}, targets::Array{Bool,2}; dataInRows::Bool=true)  = confusionMatrix2(2Float64.(outputs), targets; dataInRows=dataInRows);


# codigo del profe

# Funciones auxiliar que permite transformar una matriz de
#  valores reales con las salidas del clasificador o clasificadores
#  en una matriz de valores booleanos con la clase en la que sera clasificada
function classifyOutputs(outputs::Array{Float64,2}; dataInRows::Bool=true)
    # Miramos donde esta el valor mayor de cada instancia con la funcion findmax
    (_,indicesMaxEachInstance) = findmax(outputs, dims= dataInRows ? 2 : 1);
    # Creamos la matriz de valores booleanos con valores inicialmente a false y asignamos esos indices a true
    outputsBoolean = Array{Bool,2}(falses(size(outputs)));
    outputsBoolean[indicesMaxEachInstance] .= true;
    # Comprobamos que efectivamente cada patron solo este clasificado en una clase
    @assert(all(sum(outputsBoolean, dims=dataInRows ? 2 : 1).==1));
    return outputsBoolean;
end;

function confusionMatrix2(outputs::Array{Bool,1}, targets::Array{Bool,1})
    @assert(length(outputs)==length(targets));
    @assert(length(outputs)==length(targets));
    # Para calcular la precision y la tasa de error, se puede llamar a las funciones definidas en la practica 2
    acc         = accuracy(outputs, targets); # Precision, definida previamente en una practica anterior
    errorRate   = 1 - acc;
    recall      = mean(  outputs[  targets]); # Sensibilidad
    specificity = mean(.!outputs[.!targets]); # Especificidad
    precision   = mean(  targets[  outputs]); # Valor predictivo positivo
    NPV         = mean(.!targets[.!outputs]); # Valor predictivo negativo
    # Controlamos que algunos casos pueden ser NaN, y otros no
    @assert(!isnan(recall) && !isnan(specificity));
    precision   = isnan(precision) ? 0 : precision;
    NPV         = isnan(NPV) ? 0 : NPV;
    # Calculamos F1
    F1          = (recall==precision==0.) ? 0. : 2*(recall*precision)/(recall+precision);
    # Reservamos memoria para la matriz de confusion
    confMatrix = Array{Int64,2}(undef, 2, 2);
    # Ponemos en las filas los que pertenecen a cada clase (targets) y en las columnas los clasificados (outputs)
    #  Primera fila/columna: negativos
    #  Segunda fila/columna: positivos
    # Primera fila: patrones de clase negativo, clasificados como negativos o positivos
    confMatrix[1,1] = sum(.!targets .& .!outputs); # VN
    confMatrix[1,2] = sum(.!targets .&   outputs); # FP
    # Segunda fila: patrones de clase positiva, clasificados como negativos o positivos
    confMatrix[2,1] = sum(  targets .& .!outputs); # FN
    confMatrix[2,2] = sum(  targets .&   outputs); # VP
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
end;

confusionMatrix2(outputs::Array{Float64,1}, targets::Array{Bool,1}; threshold::Float64=0.5) = confusionMatrix2(Array{Bool,1}(outputs.>=threshold), targets);


function confusionMatrix2(outputs::Array{Bool,2}, targets::Array{Bool,2}; weighted::Bool=true)
    @assert(size(outputs)==size(targets));
    numClasses = size(targets,2);
    # Nos aseguramos de que no hay dos columnas
    @assert(numClasses!=2);
    if (numClasses==1)
        return confusionMatrix2(outputs[:,1], targets[:,1]);
    else
        # Nos aseguramos de que en cada fila haya uno y sólo un valor a true
        @assert(all(sum(outputs, dims=2).==1));
        # Reservamos memoria para las metricas de cada clase, inicializandolas a 0 porque algunas posiblemente no se calculen
        recall      = zeros(numClasses);
        specificity = zeros(numClasses);
        precision   = zeros(numClasses);
        NPV         = zeros(numClasses);
        F1          = zeros(numClasses);
        # Reservamos memoria para la matriz de confusion
        confMatrix  = Array{Int64,2}(undef, numClasses, numClasses);
        # Calculamos el numero de patrones de cada clase
        numInstancesFromEachClass = vec(sum(targets, dims=1));
        # Calculamos las metricas para cada clase, esto se haria con un bucle similar a "for numClass in 1:numClasses" que itere por todas las clases
        #  Sin embargo, solo hacemos este calculo para las clases que tengan algun patron
        #  Puede ocurrir que alguna clase no tenga patrones como consecuencia de haber dividido de forma aleatoria el conjunto de patrones entrenamiento/test
        #  En aquellas clases en las que no haya patrones, los valores de las metricas seran 0 (los vectores ya estan asignados), y no se tendran en cuenta a la hora de unir estas metricas
        for numClass in findall(numInstancesFromEachClass.>0)
            # Calculamos las metricas de cada problema binario correspondiente a cada clase y las almacenamos en los vectores correspondientes
            (_, _, recall[numClass], specificity[numClass], precision[numClass], NPV[numClass], F1[numClass], _) = confusionMatrix2(outputs[:,numClass], targets[:,numClass]);
        end;

        # Reservamos memoria para la matriz de confusion
        confMatrix = Array{Int64,2}(undef, numClasses, numClasses);
        # Calculamos la matriz de confusión haciendo un bucle doble que itere sobre las clases
        for numClassTarget in 1:numClasses, numClassOutput in 1:numClasses
            # Igual que antes, ponemos en las filas los que pertenecen a cada clase (targets) y en las columnas los clasificados (outputs)
            confMatrix[numClassTarget, numClassOutput] = sum(targets[:,numClassTarget] .& outputs[:,numClassOutput]);
        end;

        # Aplicamos las forma de combinar las metricas macro o weighted
        if weighted
            # Calculamos los valores de ponderacion para hacer el promedio
            weights = numInstancesFromEachClass./sum(numInstancesFromEachClass);
            recall      = sum(weights.*recall);
            specificity = sum(weights.*specificity);
            precision   = sum(weights.*precision);
            NPV         = sum(weights.*NPV);
            F1          = sum(weights.*F1);
        else
            # No realizo la media tal cual con la funcion mean, porque puede haber clases sin instancias
            #  En su lugar, realizo la media solamente de las clases que tengan instancias
            numClassesWithInstances = sum(numInstancesFromEachClass.>0);
            recall      = sum(recall)/numClassesWithInstances;
            specificity = sum(specificity)/numClassesWithInstances;
            precision   = sum(precision)/numClassesWithInstances;
            NPV         = sum(NPV)/numClassesWithInstances;
            F1          = sum(F1)/numClassesWithInstances;
        end;
        # Precision y tasa de error las calculamos con las funciones definidas previamente
        acc = accuracy(outputs, targets; dataInRows=true);
        errorRate = 1 - acc;

        return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
    end;
end;


function confusionMatrix2(outputs::Array{Any,1}, targets::Array{Any,1}; weighted::Bool=true)
    # Comprobamos que todas las clases de salida esten dentro de las clases de las salidas deseadas
    @assert(all([in(output, unique(targets)) for output in outputs]));
    classes = unique(targets);
    # Es importante calcular el vector de clases primero y pasarlo como argumento a las 2 llamadas a oneHotEncoding para que el orden de las clases sea el mismo en ambas matrices
    return confusionMatrix2(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted);
end;

confusionMatrix2(outputs::Array{Float64,2}, targets::Array{Bool,2}; weighted::Bool=true) = confusionMatrix2(classifyOutputs(outputs), targets; weighted=weighted);

# De forma similar a la anterior, añado estas funcion porque las RR.NN.AA. dan la salida como matrices de valores Float32 en lugar de Float64
# Con estas funcion se pueden usar indistintamente matrices de Float32 o Float64
confusionMatrix2(outputs::Array{Float32,2}, targets::Array{Bool,2}; weighted::Bool=true) = confusionMatrix2(convert(Array{Float64,2}, outputs), targets; weighted=weighted);
printConfusionMatrix2(outputs::Array{Float32,2}, targets::Array{Bool,2}; weighted::Bool=true) = printConfusionMatrix2(convert(Array{Float64,2}, outputs), targets; weighted=weighted);

# Funciones auxiliares para visualizar por pantalla la matriz de confusion y las metricas que se derivan de ella
function printConfusionMatrix2(outputs::Array{Bool,2}, targets::Array{Bool,2}; weighted::Bool=true)
    (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix2(outputs, targets; weighted=weighted);
    numClasses = size(confMatrix,1);
    writeHorizontalLine() = (for i in 1:numClasses+1 print("--------") end; println(""); );
    writeHorizontalLine();
    print("\t| ");
    if (numClasses==2)
        println(" - \t + \t|");
    else
        print.("Cl. ", 1:numClasses, "\t| ");
    end;
    println("");
    writeHorizontalLine();
    for numClassTarget in 1:numClasses
        # print.(confMatrix[numClassTarget,:], "\t");
        if (numClasses==2)
            print(numClassTarget == 1 ? " - \t| " : " + \t| ");
        else
            print("Cl. ", numClassTarget, "\t| ");
        end;
        print.(confMatrix[numClassTarget,:], "\t| ");
        println("");
        writeHorizontalLine();
    end;
    println("Accuracy: ", acc);
    println("Error rate: ", errorRate);
    println("Recall: ", recall);
    println("Specificity: ", specificity);
    println("Precision: ", precision);
    println("Negative predictive value: ", NPV);
    println("F1-score: ", F1);
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
end;
printConfusionMatrix2(outputs::Array{Float64,2}, targets::Array{Bool,2}; weighted::Bool=true) =  printConfusionMatrix2(classifyOutputs(outputs), targets; weighted=weighted)
