using Images
using FileIO

#io = open("../../aa_2021/cara_positivo/1.jpeg","r")
io = open("../../aa_2021/cara_negativo/1.jpeg","r")
img = load(io)

h_img = size(img, 1)
w_img = size(img, 2)

h_desde = floor(Int, (h_img * 0.45))
h_hasta = floor(Int, (h_img * 0.90))

w_desde = floor(Int, (w_img * 0.10))
w_hasta = floor(Int, (w_img * 0.90))

recorte = img[h_desde:h_hasta, w_desde:w_hasta]

#close(io)
