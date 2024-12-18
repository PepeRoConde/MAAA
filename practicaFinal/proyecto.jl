import Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("Statistics")
Pkg.add("StatsBase")
Pkg.add("Random")
Pkg.add("ScikitLearn")
Pkg.add("Plots")
Pkg.add("MLBase")

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

# 1 Cargar los datos y descripción
# ################################

df = CSV.read("Datos_Practica_Evaluacion_1.csv", DataFrame)

num_instancias, num_variables = size(df)
num_individuos = length(unique(df[:, 1]))
num_clases_salida = length(unique(df[:, end]))

println("Número de variables: $num_variables")
println("Número de instancias: $num_instancias")
println("Número de individuos: $num_individuos")
println("Número de clases de salida: $num_clases_salida")

# 2 Calcular porcentaje de nulos
# ##############################

num_nulos_totales = 0

for col ∈ names(df)
    num_nulos = count(ismissing, df[:, col])
    global num_nulos_totales += num_nulos
    porcentaje_nulos = (num_nulos / num_instancias) * 100
end

porcentaje_nulos_totales = (num_nulos_totales / (num_instancias * num_variables)) * 100
println("Porcentaje total de nulos en el conjunto: $porcentaje_nulos_totales%")  


# 3 Preparar los datos para las técnicas de clasificación
# #######################################################

for col ∈ names(df)
    if eltype(df[!, col]) == Union{Missing, Float64}
        df[ismissing.(df[!, col]), col] .= mean(skipmissing(df[!, col]))
        df[!, col] = Float64.(df[!, col]) 
    end
end

println("Valores nulos rellenados.")

# 4 Segmentar el 10% de los datos usando HoldOut
# ##############################################

Random.seed!(172)

holdout_individuos = shuffle(unique(df[:, :subject]))[1:Int(round(0.1 * length(unique(df[:, :subject]))))]  
holdout_df = filter(fila -> fila.subject in holdout_individuos, df)
train_df = filter(fila -> !(fila.subject in holdout_individuos), df)

println("Individuos en el holdout: ", holdout_individuos)
println("Tamaño del conjunto de entrenamiento: $(size(train_df)[1])")
println("Tamaño del conjunto de holdout: $(size(holdout_df)[1])")

# 5 Fold y escalado
# #################

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



#################################################
#################################################
#        MODELOS BASICOS
#################################################
#################################################

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


#################################################
#################################################
#        ENSEMBLES
#################################################
#################################################


#####
# 10
#####


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
