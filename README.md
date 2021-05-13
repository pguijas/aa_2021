# Prácticas Aprendizaje Automático
## Ingeniería Informática UDC - Curso 2020/2021
## Autores
* Pedro Guijas Bravo
* Eliseo Bao Souto
* Daniel Boubeta Portela
* Eduardo Pérez Fraguela
* Héctor Padín Torrente

## Uso y Generación del dataset
Para generar el archivo faces.data (ya existe en el repo), ya sea porque se quiere cambiar la extracción o por el motivo que sea, será necesario descargar el dataset con imágenes. Actualmente se aplica la extracción de características que aparece en la siguiente imagen, aun que se pueden aplicar hasta 8 extracciones diferentes.
<div style="text-align:center"><img width="40%" src="img/char_hec.jpeg"/></div>
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
| Precision | % | % | % | % | % |
| F1-Score | % | % | % | % | % |

</div>


OEEEEo

<div align="center">

| | Cara | Mascarilla | No Cara |
|:-:|:-:|:-:|:-:|
| Cara | 0 | 0 | 0 |
| Mascarilla | 0 | 0 | 0 |
| No Cara | 0 | 0 | 0 |

Matriz de confusion para --.
</div>
