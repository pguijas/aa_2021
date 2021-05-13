# Sistema de detección de mascarillas en rostros

<div style="text-align:center">

<img width="50%" src="img/UDC-Emblema.jpeg"/>

</div>
<br><br>

![Build Status](https://shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Julia Version](img/cont.jpeg)
![Julia Version](img/julia.jpeg)
![Julia Version](img/flux.jpeg)
![Julia Version](img/sktl.jpeg)
## Autores
* Pedro Guijas Bravo
* Eliseo Bao Souto
* Daniel Boubeta Portela
* Eduardo Pérez Fraguela
* Héctor Padín Torrente

## Uso y Generación del dataset
Para generar el archivo faces.data (ya existe en el repo), ya sea porque se quiere cambiar la extracción o por el motivo que sea, será necesario descargar el dataset con imágenes. Actualmente se aplica la extracción de características que aparece en la siguiente imagen, aun que se pueden aplicar hasta 8 extracciones diferentes.

<div style="text-align:center">

<img width="40%" src="img/char_hec.jpeg"/>

</div>
<br><br>

### Generación del dataset
Tenemos un archivo llamado *dataset.zip*, que contiene todas las imágenes que utilizamos en el desarrollo de la práctica. Por lo tanto, para cambiar la extracción llegaría con descomprimir este archivo en la ruta en la que se encuentra.

<code>cd dataset</code><br>
<code>unzip dataset.zip</code>

## Introducción

Para resolver el problema, utilizaremos una Redes de Neuronas Artificiales (RR.NN.AA.) *densas* y Redes de Neuronas Convolucionales (C.N.N.), Máquinas de Soporte Vectorial (SVM) o en inglés *Support Vector Machine*, k-vecinos próximos (KNN), en inglés *k-Nearest Neighbors*. En las RR.NN.AA., probaremos con distintas arquitecturas, de forma que nos quedemos con la configuración que mejores resultados obtenga y mejor se adapte al problema.

## Dataset
La Base de Datos (BD) del problema ha sido realizada por nosostros mismos. Para crear la base de datos, elegimos minuciosamente las imágenes siguiendo las pautas previamente establecidas en las restricciones, sacando las fotos de nuestros conocidos o familiares y de internet en páginas como [*This person does not exist*](https://thispersondoesnotexist.com). Cuenta con 140 imágenes de caras sin mascarillas, 140 imágenes de no-caras y 140 imágenes de caras con mascarillas.

| Cara | Cara con mascarilla | No cara |
:-:|:-:|:-:
![cara](img/ejemplo_cara.jpeg)  |  ![cara con mascarilla](img/ejemplo_mascarilla.png) | ![no cara](img/ejemplo_no_cara.jpeg)

Mencionar URL.


## Resultados
Resultados
<div align="center">

| | RNA | CNN | SVM | Árbol de Decisión | KNN |
|:-:|:-:|:-:|:-:|:-:|:-:|
| Precision | 90.58% | % | 95.63% | 82.78% | 90.36% |
| F1-Score | 90.48% | % | 95.54% | 82.60% | 90.47% |

</div>


OEEEEo

<div align="center">

| | Cara | Mascarilla | No Cara |
|:-:|:-:|:-:|:-:|
| Cara | 16 | 0 | 0 |
| Mascarilla | 0 | 12 | 0 |
| No Cara | 0 | 0 | 13 |

Matriz de confusion para RNA.

| | Cara | Mascarilla | No Cara |
|:-:|:-:|:-:|:-:|
| Cara | 16 | 0 | 0 |
| Mascarilla | 0 | 12 | 0 |
| No Cara | 0 | 0 | 13 |

Matriz de confusion para SVM con kernel gausiano.

| | Cara | Mascarilla | No Cara |
|:-:|:-:|:-:|:-:|
| Cara | 10 | 0 | 0 |
| Mascarilla | 1 | 16 | 0 |
| No Cara | 1 | 0 | 10 |

Matriz de confusion para KNN.

| | Cara | Mascarilla | No Cara |
|:-:|:-:|:-:|:-:|
| Cara | 19 | 0 | 0 |
| Mascarilla | 1 | 11 | 2 |
| No Cara | 1 | 1 | 5 |

Matriz de confusion para Árboles de Decisión.
</div>
