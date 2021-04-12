using FileIO
using Images

img = load("../datasets/cara_positivo/1.jpeg");
#display(img);
#display(img[20:50,20:70]);

# una función que nos devuelva partes de una img
function face_features(image::Array{RGB{Normed{UInt8,8}},2})::Array{RGB{Normed{UInt8,8}},2}
    (h, w) = size(image);
    eyes_and_eyebrows = image[(h ÷ 18):(h ÷ 3), (convert(Int64, round(w*0.0547)):convert(Int64, round(w*0.9423)))];
    #checkbones =
    return eyes_and_eyebrows;
end;

display(face_features(img))
