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
function loadFolderImages(folderName::String, extr::Symbol)

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
    if extr==:A1
        return imageToColorArray.(images);
    elseif extr==:A21
        return face_features_1.(images);
    elseif extr==:A22
        return face_features_2.(images);
    elseif extr==:A23
        return face_features_3.(images);
    elseif extr==:A31
        return face_features_masc.(images);
    elseif extr==:A32
        return face_features_masc2.(images);
    elseif extr==:A33
        return face_features_masc3.(images);
    elseif extr==:A34
        return face_features_masc4.(images);
    end;

end;

# =============================================================================

# Funcion para cargar todo el dataset (positivos y negativos)
function loadDataset(folderName::String, extr::Symbol)
    caraDataset = loadFolderImages(string(folderName, "/recortes"),extr);
    negativeDataset = loadFolderImages(string(folderName, "/cara_negativo"),extr);
    mascarillaDataset = loadFolderImages(string(folderName, "/cara_mascarilla"),extr);
    return (caraDataset, negativeDataset, mascarillaDataset);
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
function getInputs(path::String; extr::Symbol=:A21)
    # Obtenemos todas las fotos clasificadas en positivas y negativas
    (caraDataset, negativeDataset, mascarillaDataset) = loadDataset(path, extr);
    # Generamos la matriz de inputs y targets
    sizeFaceDataset = size(caraDataset,1);
    sizeMaskDataset = size(mascarillaDataset,1);
    sizeNegativeDataset = size(negativeDataset,1);
    rows = sizeFaceDataset + sizeMaskDataset + sizeNegativeDataset;
    if extr==:A1
        cols = 6;
    elseif extr==:A21
        cols = 42;
    elseif (extr==:A22 || extr==:A23)
        cols = 36;
    elseif (extr==:A31)
        cols = 48;
    elseif (extr==:A32)
        cols = 66;
    elseif (extr==:A33)
        cols = 72;
    elseif (extr==:A34)
        cols = 84;
    end;
    inputs = Array{Float64, 2}(undef, rows, cols);
    targets = Array{String, 1}(undef, rows);
    #@show(cols);
    #@show(rows);
    #==
    Cara || No_Cara || Cara_Masc
    ==#
    # ¿?¿?targets=convert(Array{Bool, 1},targets)
    v1 = false;
    if extr==:A1
        v1 = true;
    end;
    # Generamos la primera parte de la matriz de inputs con los elementos
    # que son positivos
    for i in 1:sizeFaceDataset
        foto = caraDataset[i];
        targets[i] = "Face";
        inputs[i,:] = v1 ? getAttributesFromImage(foto) : getAttrFromImgv2(foto);
    end;

    # Generamos la tercera parte de la matriz de inputs con los elementos
    # que son caras con mascarilla
    for i in 1:sizeMaskDataset
        foto = mascarillaDataset[i];
        targets[sizeFaceDataset+i] = "Mask";
        inputs[sizeFaceDataset+i,:] = v1 ? getAttributesFromImage(foto) : getAttrFromImgv2(foto);
    end;

    # Generamos la segunda parte de la matriz de inputs con los elementos
    # que son negativos
    aux = sizeFaceDataset+sizeMaskDataset;
    for i in 1:sizeNegativeDataset
        foto = negativeDataset[i];
        targets[aux+i] = "NotFace";
        inputs[aux+i,:] = v1 ? getAttributesFromImage(foto) : getAttrFromImgv2(foto);
    end;

    return (inputs,targets)
end

function write_dataset(file_name::String,inputs::Array{Float64, 2},targets::Array{String, 1})
    f = open(file_name, "w")
    for line in 1:size(inputs,1)
        string_line=""
        for item in inputs[line,:]
            string_line=string(string_line,item,",")
        end
        string_line=string(string_line,targets[line])
        print(f,string_line)
        println(f,"")
    end
end;


# Segunda aproximación

function getAttrFromImgv2(array)
    # Para cada imágen, sacamos las características y las juntamos todas en
    # un solo array de atributos
    attr = [];
    for image = array
        for 𝑋 = getAttributesFromImage(image)
            push!(attr, 𝑋);
        end;
    end;
    #@show(attr);
    return attr;
end;

function face_features_1(image::Array{RGB{Normed{UInt8,8}},2})::Array{Array{Float64, 3}}

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

    # ya la devolvemos el formato de Float64
    return imageToColorArray.(array_of_images);
end;

function face_features_2(image::Array{RGB{Normed{UInt8,8}},2})::Array{Array{Float64, 3}}

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
    mouth = image[(h ÷ 20 * 10):(h ÷ 20 * 16), (w ÷ 20 * 4):(w ÷ 20 * 16)];
    push!(array_of_images, mouth);

    # ya la devolvemos el formato de Float64
    return imageToColorArray.(array_of_images);
end;

function face_features_3(image::Array{RGB{Normed{UInt8,8}},2})::Array{Array{Float64, 3}}

    # devolvemos un array de imágenes
    array_of_images = [];
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

    # ya la devolvemos el formato de Float64
    return imageToColorArray.(array_of_images);
end;

function face_features_masc(image::Array{RGB{Normed{UInt8,8}},2})::Array{Array{Float64, 3}}

    # devolvemos un array de imágenes
    array_of_images = [image];
    # tamaños de la imagen para recortes
    (h, w) = size(image);
    # empieza la extracción de características
    counc = image[1:(h ÷ 20 * 5), (w ÷ 20 * 8):(w ÷ 20 * 12)];
    push!(array_of_images, counc);
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
    #visualize_masc(img, h, w);

    # ya la devolvemos el formato de Float64
    return imageToColorArray.(array_of_images);
end;

function face_features_masc2(image::Array{RGB{Normed{UInt8,8}},2})::Array{Array{Float64, 3}}

    # devolvemos un array de imágenes
    array_of_images = [image];
    # tamaños de la imagen para recortes
    (h, w) = size(image);
    # empieza la extracción de características
    eyes_and_eyebrows = image[(1):(h ÷ 3), (1):(w)];
    push!(array_of_images, eyes_and_eyebrows);
    counc = image[1:(h ÷ 20 * 5), (w ÷ 20 * 8):(w ÷ 20 * 12)];
    push!(array_of_images, counc);
    left_eye = image[(h ÷ 20):(h ÷ 3), (w ÷ 20):(w ÷ 20 * 9)];
    push!(array_of_images, left_eye);
    right_eye = image[(h ÷ 20 * 2):(h ÷ 20 * 5), (w ÷ 20 * 12):(w ÷ 20 * 18)];
    push!(array_of_images, right_eye);
    left_eye2 = image[(h ÷ 20 * 2):(h ÷ 20 * 5), (w ÷ 20 * 2):(w ÷ 20 * 8)];
    push!(array_of_images, left_eye);
    right_eye2 = image[(h ÷ 20):(h ÷ 3), (w ÷ 20 * 11):(w ÷ 20 * 19)];
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
    #visualize_masc2(img, h, w);

    # ya la devolvemos el formato de Float64
    return imageToColorArray.(array_of_images);
end;

function face_features_masc3(image::Array{RGB{Normed{UInt8,8}},2})::Array{Array{Float64, 3}}

    # devolvemos un array de imágenes
    array_of_images = [image];
    # tamaños de la imagen para recortes
    (h, w) = size(image);
    # empieza la extracción de características
    counc = image[1:(h ÷ 20 * 5), (w ÷ 20 * 8):(w ÷ 20 * 12)];
    push!(array_of_images, counc);
    left_eye = image[(h ÷ 20):(h ÷ 3), (w ÷ 20):(w ÷ 20 * 9)];
    push!(array_of_images, left_eye);
    right_eye = image[(h ÷ 20 * 2):(h ÷ 20 * 5), (w ÷ 20 * 12):(w ÷ 20 * 18)];
    push!(array_of_images, right_eye);
    left_eye2 = image[(h ÷ 20 * 2):(h ÷ 20 * 5), (w ÷ 20 * 2):(w ÷ 20 * 8)];
    push!(array_of_images, left_eye);
    right_eye2 = image[(h ÷ 20):(h ÷ 3), (w ÷ 20 * 11):(w ÷ 20 * 19)];
    push!(array_of_images, right_eye);
    left_eyebrow = image[(1):(h ÷ 20 * 2), (w ÷ 20):(w ÷ 20 * 9)];
    push!(array_of_images, left_eyebrow);
    right_eyebrow = image[(1):(h ÷ 20 * 2), (w ÷ 20 * 11):(w ÷ 20 * 19)];
    push!(array_of_images, right_eyebrow);
    left_checkb = image[(h ÷ 20 * 6):(h ÷ 20 * 12), (w ÷ 20):(w ÷ 20 * 7)];
    push!(array_of_images, left_checkb);
    right_checkb = image[(h ÷ 20 * 6):(h ÷ 20 * 12), (w ÷ 20 * 13):(w ÷ 20 * 19)];
    push!(array_of_images, right_checkb);
    nose = image[(h ÷ 20 * 6):(h ÷ 20 * 12), (w ÷ 20 * 7):(w ÷ 20 * 13)];
    push!(array_of_images, nose);
    mouth = image[(h ÷ 20 * 10):(h ÷ 20 * 16), (w ÷ 20 * 4):(w ÷ 20 * 16)];
    push!(array_of_images, mouth);
    # visualizamos los recortes
    # visualize_masc3(img, h, w);

    # ya la devolvemos el formato de Float64
    return imageToColorArray.(array_of_images);
end;


function face_features_masc4(image::Array{RGB{Normed{UInt8,8}},2})::Array{Array{Float64, 3}}

    # devolvemos un array de imágenes
    array_of_images = [image];
    # tamaños de la imagen para recortes
    (h, w) = size(image);
    # empieza la extracción de características
    eyes_and_eyebrows = image[(1):(h ÷ 3), (1):(w)];
    push!(array_of_images, eyes_and_eyebrows);
    counc = image[1:(h ÷ 20 * 5), (w ÷ 20 * 8):(w ÷ 20 * 12)];
    push!(array_of_images, counc);
    left_eye = image[(h ÷ 20):(h ÷ 3), (w ÷ 20):(w ÷ 20 * 9)];
    push!(array_of_images, left_eye);
    right_eye = image[(h ÷ 20 * 2):(h ÷ 20 * 5), (w ÷ 20 * 12):(w ÷ 20 * 18)];
    push!(array_of_images, right_eye);
    left_eye2 = image[(h ÷ 20 * 2):(h ÷ 20 * 5), (w ÷ 20 * 2):(w ÷ 20 * 8)];
    push!(array_of_images, left_eye);
    right_eye2 = image[(h ÷ 20):(h ÷ 3), (w ÷ 20 * 11):(w ÷ 20 * 19)];
    push!(array_of_images, right_eye);
    left_eyebrow = image[(1):(h ÷ 20 * 2), (w ÷ 20):(w ÷ 20 * 9)];
    push!(array_of_images, left_eyebrow);
    right_eyebrow = image[(1):(h ÷ 20 * 2), (w ÷ 20 * 11):(w ÷ 20 * 19)];
    push!(array_of_images, right_eyebrow);
    left_checkb = image[(h ÷ 20 * 6):(h ÷ 20 * 12), (w ÷ 20):(w ÷ 20 * 7)];
    push!(array_of_images, left_checkb);
    right_checkb = image[(h ÷ 20 * 6):(h ÷ 20 * 12), (w ÷ 20 * 13):(w ÷ 20 * 19)];
    push!(array_of_images, right_checkb);
    nose = image[(h ÷ 20 * 6):(h ÷ 20 * 12), (w ÷ 20 * 7):(w ÷ 20 * 13)];
    push!(array_of_images, nose);
    mouth = image[(h ÷ 20 * 10):(h ÷ 20 * 16), (w ÷ 20 * 4):(w ÷ 20 * 16)];
    push!(array_of_images, mouth);
    mouth_and_chin = image[(h ÷ 20 * 8):(h ÷ 20 * 16), (1):(w)];
    push!(array_of_images, mouth_and_chin);
    # visualizamos los recortes
    # visualize_masc3(img, h, w);

    # ya la devolvemos el formato de Float64
    return imageToColorArray.(array_of_images);
end;
