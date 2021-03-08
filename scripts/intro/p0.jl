# =============================================================================
# Eliseo Bao Souto
# Práctica 1 - Introdución
# =============================================================================

using DelimitedFiles

# -----------------------------------------------------------------------------

#=
Función para normalizar os campos categóricos en valores booleanos.

Comparamos cada elemento de x coas diferentes clases. Isto devuelve un
vector con n elementos, donde n é o número de clases diferentes. Cada
elemento corresponde a unha columna da matriz output.

Isto ten unha salvedade. Se o número de clases é 2, neste caso só temos
unha saída desexada por patrón, polo que a matriz ten tamaño Nx!, é dicir,
é un vector. Por este motivo é máis práctico que en lugar de Array{Bool,2}
sexa de tipo Array{Bool,1} (Parte true da cláusula if-else)
=#
function normalize(x::Array)

    clases = unique(x)
    if size(clases, 1) == 2
        output = [convert(Array{Bool}, x .== clases[1])]
    else
        output = [convert(Array{Bool}, x .== i) for i in clases]
    end

    return reduce(hcat, output)
end

# -----------------------------------------------------------------------------

# Cargamos a base de datos
dataset = readdlm("iris.data",',')

# Separamos os inputs dos outputs
inputs = dataset[:,1:4]
targets = dataset[:,5]

# Programación defensiva (checkeando condición)
@assert (size(inputs,1) == size(targets,1))

# Convertimos os valores dos inputs de Any a Float64
inputs = Float64.(inputs)

# Convertimos los valores dos targets de Any a String
targets = String.(targets)

# Codificamos os valores categóricos dos targets
targets = normalize(targets)

# Programación defensiva (checkeando condición)
@assert (size(inputs,1) == size(targets,1))

# para no tener que hacer más la traspuesta
inputs = convert(Array{Float64,2}, inputs')
targets = convert(Array{Bool,2}, targets')
