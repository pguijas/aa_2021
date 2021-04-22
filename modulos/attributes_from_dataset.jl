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
function loadFolderImages(folderName::String; v2=true)
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
    return v2 ? face_features_hector.(images) : imageToColorArray.(images);
end;

# =============================================================================

# Funcion para cargar todo el dataset (positivos y negativos)
function loadDataset(folderName::String; v2=true)
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

# Funcion que obtiene una matriz Nx6, donde N es el numero de elementos
# (positivos + negativos) y 6 es el numero de columnas (varianza, media y
# desviacion tipica) para cada canal RGB y un vector de targets
function getInputs(path::String; cols::Int64=42, v2::Bool=true)
    # Obtenemos todas las fotos clasificadas en positivas y negativas
    (positiveDataset, negativeDataset) = loadDataset(path);
    # Generamos la matriz de inputs y targets
    rows = size(positiveDataset,1) + size(negativeDataset,1);

    inputs = Array{Float64, 2}(undef, rows, cols);
    targets = [
        trues(size(positiveDataset,1));
        falses(size(negativeDataset,1));
    ];
    targets=convert(Array{Bool, 1},targets)

    # Generamos la primera parte de la matriz de inputs con los elementos
    # que son positivos
    for i in 1:size(positiveDataset,1)
        foto = positiveDataset[i];
        inputs[i,:] = v2 ? getAttrFromImgv2(foto) : getAttributesFromImage(foto);
    end;


    # Generamos la segunda parte de la matriz de inputs con los elementos
    # que son negativos
    for i in (size(positiveDataset,1) + 1):rows
        foto = negativeDataset[rows-i + 1];
        inputs[i,:] = v2 ? getAttrFromImgv2(foto) : getAttributesFromImage(foto);
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



# Segunda aproximación

function getAttrFromImgv2(array)
    # Array de imágenes
    #array = face_features_hector(foto);
    # Atributos para toda la imagen
    # attr = getAttributesFromImage(imageToColorArray(foto));
    #@show(inputs);
    #println()
    # Para cada imágen, sacamos las características y las juntamos todas en
    # un solo array de atributos
    attr = [];
    for image = array
        for 𝑋 = getAttributesFromImage(image)
            push!(attr, 𝑋);
        end;
    end;
    #@show(inputs);
    return attr;
end;

function face_features_hector(image::Array{RGB{Normed{UInt8,8}},2})::Array{Array{Float64, 3}}

    # devolvemos un array de imágenes
    array_of_images = [image];
    # tamaños de la imagen para recortes
    (h, w) = size(image);
    # empieza la extracción de características
    left_eye = image[(h ÷ 20):(h ÷ 3), (w ÷ 20):(w ÷ 20 * 9)];
    push!(array_of_images, left_eye);
    right_eye = image[(h ÷ 20):(h ÷ 3), (w ÷ 20 * 11):(w ÷ 20 * 19)];
    push!(array_of_images, right_eye);
    left_checkb = image[(h ÷ 20 * 6):(h ÷ 20 * 12), (w ÷ 20):(w ÷ 20 * 7)];
    push!(array_of_images, left_checkb);
    right_checkb = image[(h ÷ 20 * 6):(h ÷ 20 * 12), (w ÷ 20 * 13):(w ÷ 20 * 19)];
    push!(array_of_images, right_checkb);
    nose = image[(h ÷ 20 * 6):(h ÷ 20 * 12), (w ÷ 20 * 7):(w ÷ 20 * 13)];
    push!(array_of_images, nose);
    mouth = image[(h ÷ 20 * 10):(h ÷ 20 * 16), (w ÷ 20 * 4):(w ÷ 20 * 16)];
    push!(array_of_images, mouth);
    # visualizamos los recortes
    # visualize_hector(img, h, w);

    # ya la devolvemos el formato de Float64
    return imageToColorArray.(array_of_images);
end;
