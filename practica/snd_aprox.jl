using FileIO;
using Images;

include("../modulos/attributes_from_dataset.jl")

function visualize1(img::Array{RGB{Normed{UInt8,8}},2}, h::Int64, w::Int64)

    # ojos y cejas
    img[(h ÷ 18):(h ÷ 3),   (w ÷ 20):(w ÷ 20)]          .= RGB(1,0,0); #raya derecha
    img[(h ÷ 18):(h ÷ 18),  (w ÷ 20):(w ÷ 20 * 19)]     .= RGB(1,0,0); #raya arriba
    img[(h ÷ 18):(h ÷ 3),   (w ÷ 20 * 19):(w ÷ 20 * 19)].= RGB(1,0,0); #raya derecha
    img[(h ÷ 3):(h ÷ 3),    (w ÷ 20):(w ÷ 20 * 19)]     .= RGB(1,0,0); #raya abajo

    # nariz y pómulos
    img[(h ÷ 20 * 6):(h ÷ 20 * 10), (w ÷ 20 * 2):(w ÷ 20 * 2)]  .= RGB(0,1,0); #raya derecha
    img[(h ÷ 20 * 6):(h ÷ 20 * 6),  (w ÷ 20 * 2):(w ÷ 20 * 18)] .= RGB(0,1,0); #raya arriba
    img[(h ÷ 20 * 6):(h ÷ 20 * 10), (w ÷ 20 * 18):(w ÷ 20 * 18)].= RGB(0,1,0); #raya derecha
    img[(h ÷ 20 * 10):(h ÷ 20 * 10),(w ÷ 20 * 2):(w ÷ 20 * 18)] .= RGB(0,1,0); #raya abajo

    # boca y barbilla
    img[(h ÷ 20 * 12):(h),          (w ÷ 20 * 4):(w ÷ 20 * 4)]  .= RGB(0,0,1); #raya derecha
    img[(h ÷ 20 * 12):(h ÷ 20 * 12),(w ÷ 20 * 4):(w ÷ 20 * 16)] .= RGB(0,0,1); #raya arriba
    img[(h ÷ 20 * 12):(h),          (w ÷ 20 * 16):(w ÷ 20 * 16)].= RGB(0,0,1); #raya derecha
    img[(h):(h),                    (w ÷ 20 * 4):(w ÷ 20 * 16)] .= RGB(0,0,1); #raya abajo

    display(img);
    save("/home/hector/Downloads/char_1.jpeg", img)
end;

# una función que nos devuelva partes de una img
function face_features(image::Array{RGB{Normed{UInt8,8}},2})::Tuple{Array{RGB{Normed{UInt8,8}},2},Array{RGB{Normed{UInt8,8}},2},Array{RGB{Normed{UInt8,8}},2}}

    (h, w) = size(image);

    eyes_and_eyebrows = image[(h ÷ 18):(h ÷ 3), (w ÷ 20):(w ÷ 20 * 19)];
    checkbones_and_nose = image[(h ÷ 20 * 6):(h ÷ 20 * 10), (w ÷ 20 * 2):(w ÷ 20 * 18)];
    mouth_and_chin = image[(h ÷ 20 * 12):(h), (w ÷ 20 * 4):(w ÷ 20 * 16)];

    visualize1(img, h, w);

    return (eyes_and_eyebrows, checkbones_and_nose, mouth_and_chin);
end;

function visualize2(img::Array{RGB{Normed{UInt8,8}},2}, h::Int64, w::Int64)

    # ojo y ceja izq
    img[(h ÷ 20):(h ÷ 3), (w ÷ 20):(w ÷ 20)]         .= RGB(1,0,0); #raya derecha
    img[(h ÷ 20):(h ÷ 20),(w ÷ 20):(w ÷ 20 * 9)]     .= RGB(1,0,0); #raya arriba
    img[(h ÷ 20):(h ÷ 3), (w ÷ 20 * 9):(w ÷ 20 * 9)] .= RGB(1,0,0); #raya derecha
    img[(h ÷ 3):(h ÷ 3),  (w ÷ 20):(w ÷ 20 * 9)]     .= RGB(1,0,0); #raya abajo

    # ojo y ceja der
    img[(h ÷ 20):(h ÷ 3), (w ÷ 20 * 11):(w ÷ 20 * 11)] .= RGB(1,0,0); #raya derecha
    img[(h ÷ 20):(h ÷ 20),(w ÷ 20 * 11):(w ÷ 20 * 19)] .= RGB(1,0,0); #raya arriba
    img[(h ÷ 20):(h ÷ 3), (w ÷ 20 * 19):(w ÷ 20 * 19)] .= RGB(1,0,0); #raya derecha
    img[(h ÷ 3):(h ÷ 3),  (w ÷ 20 * 11):(w ÷ 20 * 19)] .= RGB(1,0,0); #raya abajo

    # pómulo izq
    img[(h ÷ 20 * 6):(h ÷ 20 * 12), (w ÷ 20):(w ÷ 20)]         .= RGB(0,1,0); #raya derecha
    img[(h ÷ 20 * 6):(h ÷ 20 * 6),  (w ÷ 20):(w ÷ 20 * 7)]     .= RGB(0,1,0); #raya arriba
    img[(h ÷ 20 * 6):(h ÷ 20 * 12), (w ÷ 20 * 7):(w ÷ 20 * 7)] .= RGB(0,1,0); #raya derecha
    img[(h ÷ 20 * 12):(h ÷ 20 * 12),(w ÷ 20):(w ÷ 20 * 7)]     .= RGB(0,1,0); #raya abajo

    # pómulo der
    img[(h ÷ 20 * 6):(h ÷ 20 * 12), (w ÷ 20 * 13):(w ÷ 20 * 13)] .= RGB(0,1,0); #raya derecha
    img[(h ÷ 20 * 6):(h ÷ 20 * 6),  (w ÷ 20 * 13):(w ÷ 20 * 19)] .= RGB(0,1,0); #raya arriba
    img[(h ÷ 20 * 6):(h ÷ 20 * 12), (w ÷ 20 * 19):(w ÷ 20 * 19)] .= RGB(0,1,0); #raya derecha
    img[(h ÷ 20 * 12):(h ÷ 20 * 12),(w ÷ 20 * 13):(w ÷ 20 * 19)] .= RGB(0,1,0); #raya abajo

    # nariz
    img[(h ÷ 20 * 3):(h ÷ 20 * 11), (w ÷ 20 * 7):(w ÷ 20 * 7)]   .= RGB(0,0,1); #raya derecha
    img[(h ÷ 20 * 3):(h ÷ 20 * 3),  (w ÷ 20 * 7):(w ÷ 20 * 13)]  .= RGB(0,0,1); #raya arriba
    img[(h ÷ 20 * 3):(h ÷ 20 * 11), (w ÷ 20 * 13):(w ÷ 20 * 13)] .= RGB(0,0,1); #raya derecha
    img[(h ÷ 20 * 11):(h ÷ 20 * 11),(w ÷ 20 * 7):(w ÷ 20 * 13)]  .= RGB(0,0,1); #raya abajo

    display(img);
    save("/home/hector/Downloads/char_2.jpeg", img)
end;

function complex_face_features(image::Array{RGB{Normed{UInt8,8}},2})::Tuple{Array{RGB{Normed{UInt8,8}},2},Array{RGB{Normed{UInt8,8}},2},Array{RGB{Normed{UInt8,8}},2},Array{RGB{Normed{UInt8,8}},2},Array{RGB{Normed{UInt8,8}},2}}

    (h, w) = size(image);

    left_eye = image[(h ÷ 20):(h ÷ 3), (w ÷ 20):(w ÷ 20 * 9)];
    right_eye = image[(h ÷ 20):(h ÷ 3), (w ÷ 20 * 11):(w ÷ 20 * 19)];
    left_checkb = image[(h ÷ 20 * 6):(h ÷ 20 * 12), (w ÷ 20):(w ÷ 20 * 7)];
    right_checkb = image[(h ÷ 20 * 6):(h ÷ 20 * 12), (w ÷ 20 * 13):(w ÷ 20 * 19)];
    nose = image[(h ÷ 20 * 3):(h ÷ 20 * 11), (w ÷ 20 * 7):(w ÷ 20 * 13)];

    visualize2(img, h, w);

    return (left_eye, right_eye, left_checkb, right_checkb, nose);
end;

function visualize3(img::Array{RGB{Normed{UInt8,8}},2}, h::Int64, w::Int64)

    # ojo y ceja izq
    img[(h ÷ 20):(h ÷ 3), (w ÷ 20):(w ÷ 20)]         .= RGB(1,0,0); #raya derecha
    img[(h ÷ 20):(h ÷ 20),(w ÷ 20):(w ÷ 20 * 9)]     .= RGB(1,0,0); #raya arriba
    img[(h ÷ 20):(h ÷ 3), (w ÷ 20 * 9):(w ÷ 20 * 9)] .= RGB(1,0,0); #raya derecha
    img[(h ÷ 3):(h ÷ 3),  (w ÷ 20):(w ÷ 20 * 9)]     .= RGB(1,0,0); #raya abajo

    # ojo y ceja der
    img[(h ÷ 20):(h ÷ 3), (w ÷ 20 * 11):(w ÷ 20 * 11)] .= RGB(1,0,0); #raya derecha
    img[(h ÷ 20):(h ÷ 20),(w ÷ 20 * 11):(w ÷ 20 * 19)] .= RGB(1,0,0); #raya arriba
    img[(h ÷ 20):(h ÷ 3), (w ÷ 20 * 19):(w ÷ 20 * 19)] .= RGB(1,0,0); #raya derecha
    img[(h ÷ 3):(h ÷ 3),  (w ÷ 20 * 11):(w ÷ 20 * 19)] .= RGB(1,0,0); #raya abajo

    # nariz y pómulos
    img[(h ÷ 20 * 6):(h ÷ 20 * 10), (w ÷ 20):(w ÷ 20)]  .= RGB(0,1,0); #raya derecha
    img[(h ÷ 20 * 6):(h ÷ 20 * 6),  (w ÷ 20):(w ÷ 20 * 19)] .= RGB(0,1,0); #raya arriba
    img[(h ÷ 20 * 6):(h ÷ 20 * 10), (w ÷ 20 * 19):(w ÷ 20 * 19)].= RGB(0,1,0); #raya derecha
    img[(h ÷ 20 * 10):(h ÷ 20 * 10),(w ÷ 20):(w ÷ 20 * 19)] .= RGB(0,1,0); #raya abajo

    # boca y barbilla
    #img[(h ÷ 20 * 12):(h),          (w ÷ 20 * 4):(w ÷ 20 * 4)]  .= RGB(0,0,1); #raya derecha
    #img[(h ÷ 20 * 12):(h ÷ 20 * 12),(w ÷ 20 * 4):(w ÷ 20 * 16)] .= RGB(0,0,1); #raya arriba
    #img[(h ÷ 20 * 12):(h),          (w ÷ 20 * 16):(w ÷ 20 * 16)].= RGB(0,0,1); #raya derecha
    #img[(h):(h),                    (w ÷ 20 * 4):(w ÷ 20 * 16)] .= RGB(0,0,1); #raya abajo

    display(img);
    save("/home/hector/Downloads/char_3.jpeg", img)
end;

function face_features_pedro(image::Array{RGB{Normed{UInt8,8}},2})::Array{Array{Float64, 3}}

    array_of_images = [];
    (h, w) = size(image);

    left_eye = image[(h ÷ 20):(h ÷ 3), (w ÷ 20):(w ÷ 20 * 9)];
    push!(array_of_images, left_eye);
    right_eye = image[(h ÷ 20):(h ÷ 3), (w ÷ 20 * 11):(w ÷ 20 * 19)];
    push!(array_of_images, right_eye);
    checkbones_and_nose = image[(h ÷ 20 * 6):(h ÷ 20 * 10), (w ÷ 20):(w ÷ 20 * 19)];
    push!(array_of_images, checkbones_and_nose);
    #mouth_and_chin = image[(h ÷ 20 * 12):(h), (w ÷ 20 * 4):(w ÷ 20 * 16)];
    #push!(array_of_images, mouth_and_chin);
    visualize3(img, h, w);

    return imageToColorArray.(array_of_images);
end;

function visualize_hector(img::Array{RGB{Normed{UInt8,8}},2}, h::Int64, w::Int64)

    # ojo y ceja izq
    img[(h ÷ 20):(h ÷ 3), (w ÷ 20):(w ÷ 20)]         .= RGB(1,0,0); #raya derecha
    img[(h ÷ 20):(h ÷ 20),(w ÷ 20):(w ÷ 20 * 9)]     .= RGB(1,0,0); #raya arriba
    img[(h ÷ 20):(h ÷ 3), (w ÷ 20 * 9):(w ÷ 20 * 9)] .= RGB(1,0,0); #raya derecha
    img[(h ÷ 3):(h ÷ 3),  (w ÷ 20):(w ÷ 20 * 9)]     .= RGB(1,0,0); #raya abajo

    # ojo y ceja der
    img[(h ÷ 20):(h ÷ 3), (w ÷ 20 * 11):(w ÷ 20 * 11)] .= RGB(1,0,0); #raya derecha
    img[(h ÷ 20):(h ÷ 20),(w ÷ 20 * 11):(w ÷ 20 * 19)] .= RGB(1,0,0); #raya arriba
    img[(h ÷ 20):(h ÷ 3), (w ÷ 20 * 19):(w ÷ 20 * 19)] .= RGB(1,0,0); #raya derecha
    img[(h ÷ 3):(h ÷ 3),  (w ÷ 20 * 11):(w ÷ 20 * 19)] .= RGB(1,0,0); #raya abajo

    # pómulo izq
    img[(h ÷ 20 * 6):(h ÷ 20 * 12), (w ÷ 20):(w ÷ 20)]         .= RGB(0,1,0); #raya derecha
    img[(h ÷ 20 * 6):(h ÷ 20 * 6),  (w ÷ 20):(w ÷ 20 * 7)]     .= RGB(0,1,0); #raya arriba
    img[(h ÷ 20 * 6):(h ÷ 20 * 12), (w ÷ 20 * 7):(w ÷ 20 * 7)] .= RGB(0,1,0); #raya derecha
    img[(h ÷ 20 * 12):(h ÷ 20 * 12),(w ÷ 20):(w ÷ 20 * 7)]     .= RGB(0,1,0); #raya abajo

    # pómulo der
    img[(h ÷ 20 * 6):(h ÷ 20 * 12), (w ÷ 20 * 13):(w ÷ 20 * 13)] .= RGB(0,1,0); #raya derecha
    img[(h ÷ 20 * 6):(h ÷ 20 * 6),  (w ÷ 20 * 13):(w ÷ 20 * 19)] .= RGB(0,1,0); #raya arriba
    img[(h ÷ 20 * 6):(h ÷ 20 * 12), (w ÷ 20 * 19):(w ÷ 20 * 19)] .= RGB(0,1,0); #raya derecha
    img[(h ÷ 20 * 12):(h ÷ 20 * 12),(w ÷ 20 * 13):(w ÷ 20 * 19)] .= RGB(0,1,0); #raya abajo

    # nariz
    img[(h ÷ 20 * 6):(h ÷ 20 * 12), (w ÷ 20 * 7):(w ÷ 20 * 7)]   .= RGB(0,0,1); #raya derecha
    img[(h ÷ 20 * 6):(h ÷ 20 * 6),  (w ÷ 20 * 7):(w ÷ 20 * 13)]  .= RGB(0,0,1); #raya arriba
    img[(h ÷ 20 * 6):(h ÷ 20 * 12), (w ÷ 20 * 13):(w ÷ 20 * 13)] .= RGB(0,0,1); #raya derecha
    img[(h ÷ 20 * 12):(h ÷ 20 * 12),(w ÷ 20 * 7):(w ÷ 20 * 13)]  .= RGB(0,0,1); #raya abajo

    # boca
    img[(h ÷ 20 * 10):(h ÷ 20 * 16),(w ÷ 20 * 4):(w ÷ 20 * 4)]  .= RGB(1,0,0); #raya derecha
    img[(h ÷ 20 * 10):(h ÷ 20 * 10),(w ÷ 20 * 4):(w ÷ 20 * 16)] .= RGB(1,0,0); #raya arriba
    img[(h ÷ 20 * 10):(h ÷ 20 * 16),(w ÷ 20 * 16):(w ÷ 20 * 16)].= RGB(1,0,0); #raya derecha
    img[(h ÷ 20 * 16):(h ÷ 20 * 16),(w ÷ 20 * 4):(w ÷ 20 * 16)] .= RGB(1,0,0); #raya abajo

    display(img);
    save("/home/hector/Downloads/char_hec.jpeg", img)
end;

function face_features_hector(image::Array{RGB{Normed{UInt8,8}},2})::Array{Array{Float64, 3}}

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
    # visualizamos los recortes
    visualize_hector(img, h, w);

    # ya la devolvemos el formato de Float64
    return imageToColorArray.(array_of_images);
end;

#img = load("../datasets/cara_positivo/3.jpeg");
img = load("/home/hector/Downloads/ex1.jpeg");
img = convert(Array{RGB{Normed{UInt8,8}},2},img);
#(feature1, feature2, feature3) = face_features(img);
#display(feature1)
#display(feature2)
#display(feature3)

#(feature1, feature2, feature3, feature4, feature5) = complex_face_features(img);
#display(feature1)
#display(feature2)
#display(feature3)
#display(feature4)
#display(feature5)

array = face_features_pedro(img);
display(array[1])
display(array[2])
display(array[3])

#=
array = face_features_hector(img);#face_features_pedro(img);
inputs = getAttributesFromImage(imageToColorArray(img));
@show(inputs);
println()
for image = array
    for 𝑋 = getAttributesFromImage(image)
        push!(inputs, 𝑋);
    end;
end;
@show(inputs);
=#
# podemos hacer varias funciones de loadFolderImages o añadir símbolos a la
# funcion del estilo :Char1, :Char2A, :Char2B, etc.


#=
Representación en el dataset de los inputs:

Nº    μ(R(Ojo izq)) σ(R(Ojo izq))   μ(G(Ojo izq))   σ(G(Ojo izq))   ...     Cara/No_Cara
1     0.xxxxxx      0.xxxxxx        0.xxxxxx        0.xxxx          ...     Si/No
...
238   0.xxxxxx      0.xxxxxx        0.xxxxxx        0.xxxx          ...     Si/No


=#
