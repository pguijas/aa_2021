using Images;
using FileIO;


function imageToArray(image::Array{RGB{Normed{UInt8,8}},2})
    matrix = Array{Float32, 3}(undef, size(image,1), size(image,2), 3)
    matrix[:,:,1] = convert(Array{Float32,2}, red.(image));
    matrix[:,:,2] = convert(Array{Float32,2}, green.(image));
    matrix[:,:,3] = convert(Array{Float32,2}, blue.(image));
    return matrix;
end;

imageToArray(image::Array{RGBA{Normed{UInt8,8}},2}) = imageToArray(RGB.(image));


function loadFolderImages(folderName::String, testRatio::Int64)

    # Comprobar que la foto este en formato .JPEG
    isImageExtension(fileName::String) = any(uppercase(fileName[end-4:end]) .== [".JPEG"]);
    images = [];
    testImages = [];

    count = 1;
    # Para cada fichero que detecte en la carpeta dada
    for fileName in readdir(folderName)
        if isImageExtension(fileName)
            # Leemos la imagen
            image = load(string(folderName, "/", fileName));
            # Comprobar que el archivo cargado sea una imagen en color
            if (typeof(image)==Array{RGBX{Normed{UInt8,8}},2})
                image=convert(Array{RGB{Normed{UInt8,8}},2},image);
            end;
            @assert(isa(image, Array{RGBA{Normed{UInt8,8}},2}) || isa(image, Array{RGB{Normed{UInt8,8}},2}))

            if (mod(count,testRatio)==0)
                # Añadimos la foto al array de test
                push!(testImages, image);
            else
                # Añadimos la foto al array de imagenes
                push!(images, image);
            end;
            count += 1;
        end;
    end;

    # Devolvemos el array con las imagenes convertidas a Float32
    return (imageToArray.(images), imageToArray.(testImages));
end;


function loadDataset(folderName::String, testRatio::Int64)
    (caras, testCaras) = loadFolderImages(string(folderName, "/DL/caras"), testRatio);
    (negativo, testNegativo) = loadFolderImages(string(folderName, "/DL/no_caras"), testRatio);
    (mascarillas, testMascarillas) = loadFolderImages(string(folderName, "/DL/mascarilla"), testRatio);
    return (caras, testCaras, negativo, testNegativo, mascarillas, testMascarillas);
end;


function getInputs(path::String, testRatio::Int64)
    # Obtenemos las matrices de entrenamiento y test
    (caras, testCaras, negativo, testNegativo,
        mascarillas, testMascarillas) = loadDataset(path, testRatio);
    sizeFaceDataset = size(caras,1);
    sizeMaskDataset = size(mascarillas,1);
    sizeNegativeDataset = size(negativo,1);
    length = sizeFaceDataset + sizeMaskDataset + sizeNegativeDataset;

    sizeFaceDatasetTest = size(testCaras,1);
    sizeMaskDatasetTest = size(testMascarillas,1);
    sizeNegativeDatasetTest = size(testNegativo,1);
    testLength = sizeFaceDatasetTest + sizeMaskDatasetTest + sizeNegativeDatasetTest;

    #train_imgs = Array{Float32, 4}(undef, size(image,1), size(image,2), 3, length);
    train_imgs = [];
    train_labels = Array{Int64, 1}(undef, length);
    #test_imgs = Array{Float32, 4}(undef, size(image,1), size(image,2), 3, testLength);
    test_imgs = [];
    test_labels = Array{Int64, 1}(undef, testLength);

    # Generamos la primera parte de la matriz de inputs con los elementos
    # que son positivos
    for i in 1:sizeFaceDataset
        train_labels[i] = 0;
        push!(train_imgs, caras[i]);
    end;

    # Generamos la tercera parte de la matriz de inputs con los elementos
    # que son caras con mascarilla
    for i in 1:sizeMaskDataset
        train_labels[sizeFaceDataset+i] = 1;
        push!(train_imgs, mascarillas[i]);
    end;

    # Generamos la segunda parte de la matriz de inputs con los elementos
    # que son negativos
    aux = sizeFaceDataset+sizeMaskDataset;
    for i in 1:sizeNegativeDataset
        train_labels[aux+i] = 2;
        push!(train_imgs, negativo[i]);
    end;

    # Generamos la primera parte de la matriz de inputs con los elementos
    # que son positivos
    for i in 1:sizeFaceDatasetTest
        test_labels[i] = 0;
        push!(test_imgs, testCaras[i]);
    end;

    # Generamos la tercera parte de la matriz de inputs con los elementos
    # que son caras con mascarilla
    for i in 1:sizeMaskDatasetTest
        test_labels[sizeFaceDatasetTest+i] = 1;
        push!(test_imgs, testMascarillas[i]);
    end;

    # Generamos la segunda parte de la matriz de inputs con los elementos
    # que son negativos
    aux = sizeFaceDatasetTest+sizeMaskDatasetTest;
    for i in 1:sizeNegativeDatasetTest
        test_labels[aux+i] = 2;
        push!(test_imgs, testNegativo[i]);
    end;

    return (train_imgs, train_labels, test_imgs, test_labels);
end


(train_imgs, train_labels, test_imgs, test_labels) = getInputs("datasets",6);
@assert(size(train_imgs,1)==size(train_labels,1))
@assert(size(test_imgs,1)==size(test_labels,1))
