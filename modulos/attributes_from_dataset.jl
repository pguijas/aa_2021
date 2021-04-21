using Images
using FileIO
using JLD2
using Statistics: std, mean, var

# =============================================================================

# Funcion para convertir una foto (en RGB) en un Array de Float64
function imageToColorArray(image::Array{RGB{Normed{UInt8,8}},2})
    matrix = Array{Float64, 3}(undef, size(image,1), size(image,2), 3)
    matrix[:,:,1] = convert(Array{Float64,2}, red.(image));
    matrix[:,:,2] = convert(Array{Float64,2}, green.(image));
    matrix[:,:,3] = convert(Array{Float64,2}, blue.(image));
    return matrix;
end;

# Funcion para convertir una foto (en RGBA) en un Array de Float64
imageToColorArray(image::Array{RGBA{Normed{UInt8,8}},2}) = imageToColorArray(RGB.(image));

# =============================================================================

# Funcion para leer todas las imagenes de una carpeta dada
function loadFolderImages(folderName::String)
    # Comprobar que la foto este en formato .JPEG
    isImageExtension(fileName::String) = any(uppercase(fileName[end-4:end]) .== [".JPEG"]);
    images = [];

    # Para cada fichero que detecte en la carpeta dada
    for fileName in readdir(folderName)
        println(fileName)
        if isImageExtension(fileName)
            # Leemos la imagen
            image = load(string(folderName, "/", fileName));
            # Comprobar que el archivo cargado sea una imagen en color
            if (typeof(image)==Array{RGBX{Normed{UInt8,8}},2})
                image=convert(Array{RGB{Normed{UInt8,8}},2},image)
            end
            @assert(isa(image, Array{RGBA{Normed{UInt8,8}},2}) || isa(image, Array{RGB{Normed{UInt8,8}},2}))
            # Añadimos la foto al array de imagenes
            push!(images, image);
        end;
    end;

    # Devolvemos el array con las imagenes convertidas a Float64
    return imageToColorArray.(images);
end;

# =============================================================================

# Funcion para cargar todo el dataset (positivos y negativos)
function loadDataset(folderName::String)
    positiveDataset = loadFolderImages(string(folderName, "/recortes"));
    negativeDataset = loadFolderImages(string(folderName, "/cara_negativo"));
    return (positiveDataset, negativeDataset);
end;

# =============================================================================

# Funcion que obtiene la varianza, desviacion y media de los 3 canales RGB de
# una imagen dada
function getAttributesFromImage(foto)
    inputs = [];
    for canal in 1:3
        mean_image = mean(foto[:,:,canal]);
        std_image = std(foto[:,:,canal]);

        push!(inputs, mean_image)
        push!(inputs, std_image)
    end;
    return inputs;
end

# =============================================================================

# Funcion que obtiene una matriz Nx9, donde N es el numero de elementos
# (positivos + negativos) y 9 es el numero de columnas (varianza, media y
# desviacion tipica) para cada canal RGB y un vector de targets
function getInputs(path)
    # Obtenemos todas las fotos clasificadas en positivas y negativas
    (positiveDataset, negativeDataset) = loadDataset(path);
    # Generamos la matriz de inputs y targets
    rows = size(positiveDataset,1) + size(negativeDataset,1);
    cols = 6;

    inputs = Array{Float64, 2}(undef, rows, cols);
    targets = [
        trues(size(positiveDataset,1));
        falses(size(negativeDataset,1))
    ];
    targets=convert(Array{Bool, 1},targets)

    # Generamos la primera parte de la matriz de inputs con los elementos
    # que son positivos
    for i in 1:size(positiveDataset,1)
        foto = positiveDataset[i];
        inputs[i,:] = getAttributesFromImage(foto);
    end;

    # Generamos la segunda parte de la matriz de inputs con los elementos
    # que son negativos
    for i in (size(positiveDataset,1) + 1):rows
        foto = negativeDataset[rows-i + 1];
        inputs[i,:] = getAttributesFromImage(foto);
    end;
    return (inputs,targets)
end

function write_dataset(file_name::String,inputs::Array{Float64, 2},targets::Array{Bool, 1})
    f = open(file_name, "w")
    for line in 1:size(inputs,1)
        string_line=""
        for item in inputs[line,:]
            string_line=string(string_line,item,",")
        end
        if (targets[line])
            string_line=string(string_line,"1")
        else
            string_line=string(string_line,"0")
        end
        print(f,string_line)
        println(f,"")
    end
end
