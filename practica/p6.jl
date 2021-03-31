# antes de hacer esto importante descargarlo para python
# pip3 install -U scikit-learn
# Reiniciamos Julia con CTR+D + ENTER
# SI ejecutamos el archivo debería ir todo bien
import Pkg;
Pkg.add("ScikitLearn")
Pkg.update()
using ScikitLearn;
@sk_import svm: SVC

model = SVC(kernel="rbf", degree=3, gamma=2, C=1);
