# %%
"""
Aldao Amoedo, Héctor

Cabaleiro Pintos, Laura

Cotardo Valcárcel, Donato José

Romero Conde, José
"""

# %%
"""
---
"""

# %%
"""
##   _Librerias_
"""

# %%
import Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("Statistics")
Pkg.add("StatsBase")
Pkg.add("Random")
Pkg.add("ScikitLearn")
Pkg.add("Plots")
Pkg.add("MLBase")

# %%
using CSV
using DataFrames
using Plots
using Statistics
using Random
using ScikitLearn
using ScikitLearn.CrossValidation: cross_val_score, KFold
using ScikitLearn.Pipelines: Pipeline, named_steps, FeatureUnion
using ScikitLearn.GridSearch: GridSearchCV 
@sk_import decomposition: (PCA, FastICA)
@sk_import discriminant_analysis: LinearDiscriminantAnalysis
@sk_import ensemble: (AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier)
@sk_import feature_selection: (SelectKBest, f_classif, mutual_info_classif, RFE)
@sk_import impute: SimpleImputer
@sk_import linear_model: LogisticRegression
@sk_import manifold: (LocallyLinearEmbedding, Isomap)
@sk_import neighbors: KNeighborsClassifier
@sk_import neural_network: MLPClassifier
@sk_import preprocessing: MinMaxScaler
@sk_import svm: SVC

# %%
"""
## Preparación de los datos
"""

# %%
"""
El punto 1, está completado, aunque creo que el 1.3, donde preparamos los datos transformandolos y rellanando los nulos está mal.
"""

# %%
"""
### 1. Cargar los datos y descripción
"""

# %%
df = CSV.read("Datos_Practica_Evaluacion_1.csv", DataFrame)

num_instancias, num_variables = size(df)
num_individuos = length(unique(df[:, 1]))
num_clases_salida = length(unique(df[:, end]))

println("Número de variables: $num_variables")
println("Número de instancias: $num_instancias")
println("Número de individuos: $num_individuos")
println("Número de clases de salida: $num_clases_salida")

# %%
"""
Los datos dados ya están cargados, y como podemos observar tienen:
* 563 Variables
* 10299 Instancias
* 30 Individuos
* 6 Clases de salida
"""

# %%
"""
### 2. Calcular porcentaje de nulos
"""

# %%
num_nulos_totales = 0

for col ∈ names(df)
    num_nulos = count(ismissing, df[:, col])
    num_nulos_totales += num_nulos
    porcentaje_nulos = (num_nulos / num_instancias) * 100
end

porcentaje_nulos_totales = (num_nulos_totales / (num_instancias * num_variables)) * 100
println("Porcentaje total de nulos en el conjunto: $porcentaje_nulos_totales%")  

# %%
"""
### 3. Preparar los datos para las técnicas de clasificación
"""

# %%
"""
Para rellenar valores faltantes tenemos que hacernos una idea de que tipo de datos encontraremos.
"""

# %%
for col ∈ names(df)
    if eltype(df[!, col]) == Union{Missing, Float64}
        df[ismissing.(df[!, col]), col] .= mean(skipmissing(df[!, col]))
        df[!, col] = Float64.(df[!, col]) 
    end
end

println("Valores nulos rellenados.")

# %%
Set([eltype(df[!,col]) for col in names(df)])  # ya no hay missing

# %%
"""
### 4. Segmentar el 10% de los datos usando HoldOut
"""

# %%
Random.seed!(172)

holdout_individuos = shuffle(unique(df[:, :subject]))[1:Int(round(0.1 * length(unique(df[:, :subject]))))]  
holdout_df = filter(fila -> fila.subject in holdout_individuos, df)
train_df = filter(fila -> !(fila.subject in holdout_individuos), df)

println("Individuos en el holdout: ", holdout_individuos)
println("Tamaño del conjunto de entrenamiento: $(size(train_df)[1])")
println("Tamaño del conjunto de holdout: $(size(holdout_df)[1])")

# %%
"""
### 5. Fold y escalado
"""

# %%
# separacion X Y
train = Array(train_df)
test = Array(holdout_df)

X_train = train[:,1:end-1]
Y_train = train[:,end]
X_test = test[:,1:end-1]
Y_test = test[:,end];
# k fold
especificacionCV = ScikitLearn.CrossValidation.KFold(size(train)[1], n_folds=5)
#folds = [(train[indicesTrain,:], train[indicesTest,:]) for (indicesTrain, indicesTest) ∈ especificacionCV]
# escalado
X_train = fit_transform!(MinMaxScaler(), X_train);
X_test = fit_transform!(MinMaxScaler(), X_test);
#folds = [((fit_transform!(MinMaxScaler(), train[:,1:end-1]),Vector(train[:,end])), (fit_transform!(MinMaxScaler(), test[:,1:end-1]),Vector(test[:,end]))) for (train, test) ∈ folds ];

# de esta forma:
# folds tiene los 5 folds (escalado cada uno independientemente)
# folds[1] es un fold
# folds[1][1] es el entrenamiento de ese fold
# folds[1][1][1] es el X del entrenamiento de ese fold

# %%
"""
---
"""

# %%
"""
# Modelos Basicos
"""

# %%
function  buenosDias(df)
    clasificadores = unique(df[!,:clasificador])
    println("MEJORES RESULTADOS POR CLASIFICADOR:")
    for nombreClasificador in clasificadores
        df_clasificador = filter(fila -> fila.clasificador == nombreClasificador, df)
        #println(df_clasificador)
        indice = argmax(df_clasificador[!,Symbol("Accuracy")])
        println(Array(df_clasificador[indice,:]))
    end
end

# %%
"""
---
"""

# %%
function plot_transformed_data(name::String, reductor, X, y)

    X_reducida = fit_transform!(reducer, X, y)
    plot = scatter(X_reduced[:, 1], X_reduced[:, 2], group=y, legend=:topright, title=name,
                xlabel="Componente 1", ylabel="Componente 2", markersize=5)
    return plot
end

# %%
resultadosModelosBasicos = DataFrame(filtrado = String[], reduccion = String[], clasificador = String[], Accuracy = Float64[])
plot_distribucion = @layout [a b c; d e]

filtrado = Dict(
   #"nada" => "passthrough",
   "anova" => SelectKBest(score_func=f_classif),
   "mi" => SelectKBest(score_func=mutual_info_classif),
   "rfe" => RFE(LogisticRegression(max_iter=10),step=0.5)
 )

reduccion = Dict(
   #"nada" => "passthrough",
   "pca" => PCA(),
   "lda" => LinearDiscriminantAnalysis(),
   "ica" => FastICA(),
   #"isomap" => Isomap(n_neighbors=25),
   #"lle" => LocallyLinearEmbedding(),
 )

clasificacion = Dict(
    "mlp" => [MLPClassifier(max_iter=10), Dict(:classifier__hidden_layer_sizes => [[50], [100], [100, 50]])],
    "knn" => [KNeighborsClassifier(), Dict(:classifier__n_neighbors =>[1, 10, 20])],
    "svm" => [SVC(), Dict(:classifier__C =>[0.1, 1, 10])]
)

for (nombreFiltro, filtro) in filtrado
    for (nombreReduccion, reduccion) in reduccion
        for (nombreClasificador, valor) in clasificacion

            clasificador = valor[1]
            parametros = valor[2]
            modelo = Pipeline([
                ("filtro", filtro),
                ("reduccion", reduccion),
                ("classifier", clasificador) 
            ])

            busqueda = GridSearchCV(modelo, parametros, cv=especificacionCV)
            fit!(busqueda, X_train, Y_train)
            mejorModelo = busqueda.best_estimator_
            mejoresParametros = busqueda.best_params_
            accuracy = busqueda.best_score_
            Y_pred = predict(mejorModelo, X_test)
            accuracy = sum(Y_pred .== Y_test) / length(Y_test)
            push!(resultadosModelosBasicos, (nombreFiltro, nombreReduccion, nombreClasificador, accuracy))
            println("Filtrado: $nombreFiltro,\nReducción: $nombreReduccion,\nClasificador: $nombreClasificador,\nparámetros: $(mejoresParametros),\nPrecisión: $accuracy\n")
        
        end
    end
end

    

buenosDias(resultadosModelosBasicos)

# %%
"""
---
"""

# %%
"""
# Ensembles
"""

# %%
"""
---
"""

# %%
"""
10. Adicionalmente, con los datos sólo con el tratamiento de Filtrado ANOVA, recrear
las siguientes técnicas
 - BaggingClassifier con clasificador base KNN con número de vecinos 5 y
número de estimadores 10 y 50
 -  AdaBoosting con estimadores SVM con kernel lineal siendo el número de
estimadores 5.
 -  GBM (GradientBoostingClasifier), con 50 estimadores y un learning_rate de
0.2
"""

# %%
# ejercicio 10

resultadosEjercicio10 = DataFrame(clasificador = String[],  Accuracy = Float64[])

clasificadoresEjercicio10 = Dict(
    "bagging10" => BaggingClassifier(estimator=KNeighborsClassifier(n_neighbors=5), n_estimators=10),
    "bagging50" => BaggingClassifier(estimator=KNeighborsClassifier(n_neighbors=5), n_estimators=50),
    "adaboosting" => AdaBoostClassifier(estimator=SVC(kernel="linear"), algorithm="SAMME", n_estimators=5),
    "gbm" => GradientBoostingClassifier(learning_rate=0.2, n_estimators=50)
    )

for (nombreClasificador, clasificador) in clasificadoresEjercicio10
    modelo = Pipeline([
        ("filtro", SelectKBest(score_func=f_classif)),
        ("classifier", clasificador)
    ])

    fit!(modelo, X_train, Y_train)
    Y_pred = predict(modelo, X_test)
    accuracy = sum(Y_pred .== Y_test) / length(Y_test)
    push!(resultadosEjercicio10, (nombreClasificador, accuracy))
    println("Clasificador: $nombreClasificador Precisión: $accuracy")
end

buenosDias(resultadosEjercicio10)

# %%
buenosDias(resultadosEjercicio10)

# %%
"""
---
"""

# %%
"""
11. Entrenar con el conjunto completo de entrenamiento (todo lo que componía el 5-
fold cross-validation) y testear son el 10% reservado
 - Coger las 5 mejores combinaciones de los modelos anteriores de
clasificación, (1 KNN, 1 SVM, 1 MLP, 1 Bagging y 1 AdaBoosting)
 - Crear un Random Forest con valor para los estimadores del 500 y
profundidad máxima de 10
 - Crear un Hard Voting con las mejores combinaciones del KNN, SVM y MLP
(uno para cada una de las técnicas)
 - Crear un Soft Voting con las mejores combinaciones del KNN, SVM y MLP
(uno para cada una de las técnicas) para los pesos coger el porcentaje de
acierto en test de cada una de las combinaciones en el 5-fold cross-
valiadation
 - Crear un Ensemble Stacking con MLP como clasificador final, así mismo,
use como base las mejores combinaciones del SVM, KNN y MLP
 - Crear un XGBoost con los valores por defecto
 - Crear un LightGBM, con los valores por defecto
 - Crear un Catboost, con los valores por defecto
"""

# %%


# %%
"""
### 6. Normalizar usando MinMaxScaler
"""

# %%
"""
---
"""

# %%
"""
## Creación de los modelos básicos
"""

# %%
"""
### 7. Filtrado
"""

# %%
"""
#### 7.1 ANOVA
"""

# %%
"""
#### 7.2 Mutual Information
"""

# %%
"""
#### 7.3 RFE con el método de LogisticRegression con una eliminación del 50% de las variables en cada pasada.
"""

# %%
"""
### 8. Reducción dimensionalidad
"""

# %%
"""
#### 8.1 PCA
"""

# %%
"""
#### 8.2 LDA
"""

# %%
"""
#### 8.3 ICA
"""

# %%
"""
#### 8.4 Isomap
"""

# %%
"""
#### 8.5 LLE
"""

# %%
"""
### 9. Clasificadores
"""

# %%
"""
#### 9.1 MLP con al menos las siguientes arquitecturas: [50], [100] [100, 50]
"""

# %%
"""
#### 9.2 KNN con valores de vecindario entre 1, 10 y 20
"""

# %%
"""
#### 9.3 SVM con el parámetro C con valores 0.1, 0.5 y 1.0
"""

# %%
"""
---
"""

# %%
"""
## Creación de los modelos ensemble
"""

# %%
"""
---
"""

# %%
"""
## Conclusiones
"""