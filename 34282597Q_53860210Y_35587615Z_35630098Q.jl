

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 1 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using FileIO, Images, JLD2, DelimitedFiles, Flux, StatsBase



function fileNamesFolder(folderName::String, extension::String)
    # Convertir la extensión a mayúsculas
    extension = uppercase(extension)

    # Filtrar los archivos que terminen con la extensión proporcionada (sin punto)
    fileNames = filter(f -> endswith(uppercase(f), ".$extension"), readdir(folderName))
    
    # Eliminar la extensión de los nombres de archivo sin usar bucles
    fileNamesWithoutExt = map(f -> splitext(f)[1], fileNames)
    
    return fileNamesWithoutExt
end;



using DelimitedFiles

function loadDataset(datasetName::String, datasetFolder::String; datasetType::DataType=Float32)
    # Construir la ruta completa al archivo con extensión .tsv
    datasetPath = joinpath(datasetFolder, datasetName * ".tsv")
    
    # Verificar si el archivo existe
    if !isfile(datasetPath)
        return nothing  # Devolver `nothing` si el archivo no existe
    end
    
    # Leer el archivo como una matriz
    data = readdlm(datasetPath, '\t')  # Leemos el archivo .tsv con delimitador de tabulación
    
    # La primera fila contiene los encabezados
    headers = data[1, :]
    
    # Buscar la columna de salida que tenga el encabezado "target"
    target_col = findall(x -> x == "target", headers)
    
    if isempty(target_col)
        return nothing  # Si no hay una columna "target", devolver `nothing`
    end
    
    # Tomar el índice de la columna de target
    target_idx = target_col[1]
    
    # Extraer las entradas (todas las columnas menos la de target)
    inputs = data[2:end, setdiff(1:size(data, 2), target_idx)]
    inputs = Array{datasetType}(inputs)  # Convertir las entradas al tipo datasetType
    
    # Extraer las salidas (la columna target) y convertirlas a booleano
    targets = data[2:end, target_idx]
    targets = targets .== 1  # Convertimos los valores {0,1} a booleanos {false, true}
    
    # Devolver las entradas y las salidas como una tupla (inputs, targets)
    return (inputs, targets)
end;



function loadImage(imageName::String, datasetFolder::String; datasetType::DataType=Float32, resolution::Int=128)

    #Añadir extensión al nombre del archivo
    imageName = imageName * ".tif"
    
    #Verificar que el archivo existe
    if !isfile(joinpath(datasetFolder, imageName))
        return nothing
    end
    
    #Cargar la imagen
    image=load(joinpath(datasetFolder, imageName))
    
    #Modificar la resolución de la imagen
    image=imresize(image, (resolution,resolution))
    
    #Convertir la imagen a una matriz
    image=gray.(image)
    
    #Cambiar el tipo de la matriz
    #image=convert(Array{datasetType,2}, image)
    image=datasetType.(image)

    return image
    
end;


function convertImagesNCHW(imageVector::Vector{<:AbstractArray{<:Real,2}})
    imagesNCHW = Array{eltype(imageVector[1]), 4}(undef, length(imageVector), 1, size(imageVector[1],1), size(imageVector[1],2));
    for numImage in Base.OneTo(length(imageVector))
        imagesNCHW[numImage,1,:,:] .= imageVector[numImage];
    end;
    return imagesNCHW;
end;


function loadImagesNCHW(datasetFolder::String; datasetType::DataType=Float32, resolution::Int=128)
    
    image = fileNamesFolder.(datasetFolder,"tif")

    imageLoad = loadImage.(image, datasetFolder, datasetType=datasetType, resolution=resolution)

    imagesNCHW = convertImagesNCHW(imageLoad)
    
    return imagesNCHW

end;




showImage(image      ::AbstractArray{<:Real,2}                                      ) = display(Gray.(image));
showImage(imagesNCHW ::AbstractArray{<:Real,4}                                      ) = display(Gray.(     hcat([imagesNCHW[ i,1,:,:] for i in 1:size(imagesNCHW ,1)]...)));
showImage(imagesNCHW1::AbstractArray{<:Real,4}, imagesNCHW2::AbstractArray{<:Real,4}) = display(Gray.(vcat(hcat([imagesNCHW1[i,1,:,:] for i in 1:size(imagesNCHW1,1)]...), hcat([imagesNCHW2[i,1,:,:] for i in 1:size(imagesNCHW2,1)]...))));



function loadMNISTDataset(datasetFolder::String; labels::AbstractArray{Int,1}=0:9, datasetType::DataType=Float32)
    
    
    dataset = load(joinpath(datasetFolder, "MNIST.jld2"))

    inputsTrain = convert.(Matrix{datasetType}, dataset["train_imgs"])
    targetsTrain = dataset["train_labels"]
    inputsTest = convert.(Matrix{datasetType}, dataset["test_imgs"])
    targetsTest = dataset["test_labels"]

    if -1 in labels
        targetsTrain[.!in.(targetsTrain, [setdiff(labels, -1)])] .= -1
        targetsTest[.!in.(targetsTest, [setdiff(labels, -1)])] .= -1
    end

    indicesTrain = in.(targetsTrain, [labels])
    indicesTest = in.(targetsTest, [labels])

    imagesTrain = convertImagesNCHW(inputsTrain[indicesTrain])
    imagesTest = convertImagesNCHW(inputsTest[indicesTest])

    return (imagesTrain, targetsTrain[indicesTrain], imagesTest, targetsTest[indicesTest])

end;


function intervalDiscreteVector(data::AbstractArray{<:Real,1})
    # Ordenar los datos
    uniqueData = sort(unique(data));
    # Obtener diferencias entre elementos consecutivos
    differences = sort(diff(uniqueData));
    # Tomar la diferencia menor
    minDifference = differences[1];
    # Si todas las diferencias son multiplos exactos (valores enteros) de esa diferencia, entonces es un vector de valores discretos
    isInteger(x::Float64, tol::Float64) = abs(round(x)-x) < tol
    return all(isInteger.(differences./minDifference, 1e-3)) ? minDifference : 0.
end



function cyclicalEncoding(data::AbstractArray{<:Real,1})
    # Encuentra el valor de m
    m = intervalDiscreteVector(data)

    
    # Normalizar los datos al intervalo [0, 2π)
    data_normalizada = 2 * π * (data .- minimum(data)) ./ (maximum(data) - minimum(data) + m)
    
    # Calcular los senos y cosenos
    valores_seno = sin.(data_normalizada)
    valores_coseno = cos.(data_normalizada)
    
    return (valores_seno, valores_coseno)
end


function loadStreamLearningDataset(datasetFolder::String; datasetType::DataType=Float32)
    # Construimos las rutas absolutas de los archivos
    data_path = joinpath(datasetFolder, "elec2_data.dat")
    label_path = joinpath(datasetFolder, "elec2_label.dat")

    # Cargamos los archivos
    data = readdlm(data_path)
    labels = readdlm(label_path)

    # Eliminamos las columnas 1 (date) y 4 (nswprice)
    data = data[:, setdiff(1:size(data, 2), [1, 4])]

    # Codificación cíclica de la primera columna (día)
    days = data[:, 1]
    sin_vals, cos_vals = cyclicalEncoding(days)

    # Eliminamos la primera columna (día) del dataset
    data = data[:, 2:end]

    # Concatenamos los valores de seno y coseno al inicio de la matriz de datos
    inputs = hcat(sin_vals, cos_vals, data)

    # Convertimos los labels a booleanos y los pasamos a un vector
    outputs = vec(Bool.(labels))

    # Cambiamos el tipo de dato de las entradas si se especifica un datasetType
    inputs = convert(Matrix{datasetType}, inputs)

    # Retornamos la tupla (inputs, outputs)
    return inputs, outputs
end

#a = loadStreamLearningDataset("datasets")

# println(a)

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Flux

indexOutputLayer(ann::Chain) = length(ann) - (ann[end]==softmax);

function newClassCascadeNetwork(numInputs::Int, numOutputs::Int)
    
    ann = Chain()
    numInputsLayer = numInputs
    
    if numOutputs > 2
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs), softmax)
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, σ))
    end
    
    return ann

end;

function addClassCascadeNeuron(previousANN::Chain; transferFunction::Function=identity)

    # Referenciar la capa de salida y las capas previas
    outputLayer =    previousANN[   indexOutputLayer(previousANN)   ]
    previousLayers = previousANN[1:(indexOutputLayer(previousANN)-1)]
    
    # Obtener el número de entradas y salidas de la capa de salida
    numInputsOutputLayer  = size(outputLayer.weight, 2)
    numOutputsOutputLayer = size(outputLayer.weight, 1)
    
    
    # Crear la nueva RNA con las capas anteriores, la nueva capa y la capa de salida
    newANN = Chain(
        previousLayers...,  # Capas anteriores
        SkipConnection(Dense(numInputsOutputLayer, 1, transferFunction), (mx, x) -> vcat(x, mx)),  # Crear la nueva capa con la neurona en cascada
        Dense(numInputsOutputLayer + 1, numOutputsOutputLayer, outputLayer.σ)  # Nueva capa de salida
        # Dense(num de entradas a la capa,num de salidas de la capa, funcion de activacion)
        )
    
    # Copiar los pesos y biases de la capa de salida de la red anterior a la nueva
    newANN_outputLayer = newANN[end]
    
    # Modificar la matriz de pesos y el vector de bias
    newANN_outputLayer.weight[:, 1:numInputsOutputLayer] .= outputLayer.weight
    newANN_outputLayer.weight[:, numInputsOutputLayer+1] .= 0.0  # Poner a 0 la última columna (nueva neurona) para que la nueva neurona no tenga impacto en las salidas originales
    newANN_outputLayer.bias .= outputLayer.bias  # Copiar el vector de bias
    
    return newANN
end;


using Flux

function trainClassANN!(ann::Chain, trainingDataset::Tuple{AbstractArray{<:Real, 2}, AbstractArray{Bool, 2}}, trainOnly2LastLayers::Bool;
    maxEpochs::Int = 1000, minLoss::Real = 0.0, learningRate::Real = 0.001, minLossChange::Real = 1e-7, lossChangeWindowSize::Int = 5)
    
    inputs, targets = trainingDataset

    # Convertir entradas a Float32 si es necesario
    if eltype(inputs) <: Float64
        inputs = Float32.(inputs)
    end

    # Inicializar optimizador Adam
    opt_state = Flux.setup(Adam(learningRate), ann)

    # Definir la función de pérdida
    loss(x, y) = (size(y, 1) == 1) ? Flux.binarycrossentropy(ann(x), y) : Flux.crossentropy(ann(x), y)

    # Almacenar valores de pérdida a lo largo del entrenamiento
    loss_history = Float64[]

    # Calcular la pérdida inicial
    current_loss = loss(inputs, targets)
    push!(loss_history, current_loss)
    println("Epoch 0: loss = ", current_loss)

    # Si se desea entrenar solo las dos últimas capas, congelar las capas anteriores
    if trainOnly2LastLayers
        for layer in 1:(length(ann) - 2)
            # Manejar capas compuestas
            if typeof(ann[layer]) <: Flux.Chain
                for sublayer in ann[layer]
                    Flux.freeze!(sublayer)
                end
            else
                Flux.freeze!(ann[layer])
            end
        end
    end

    # Bucle de entrenamiento
    for epoch in 1:maxEpochs
        # Realizar un ciclo de entrenamiento
        gs = Flux.gradient(() -> loss(inputs, targets), Flux.params(ann))
        Flux.update!(opt_state, ann, gs)

        # Calcular la nueva pérdida
        current_loss = loss(inputs, targets)
        push!(loss_history, current_loss)

        # Imprimir la pérdida actual
        println("Epoch $epoch: loss = ", current_loss)

        # Verificar el criterio de parada por pérdida mínima
        if current_loss <= minLoss
            println("Criterio de parada alcanzado: minLoss alcanzado en el ciclo $epoch.")
            break
        end

        # Verificar el criterio de parada por cambio en la pérdida
        if length(loss_history) > lossChangeWindowSize
            lossWindow = loss_history[end - lossChangeWindowSize + 1:end]
            minLossValue, maxLossValue = extrema(lossWindow)
            if ((maxLossValue - minLossValue) / minLossValue <= minLossChange)
                println("Criterio de parada alcanzado: cambio en la pérdida menor al mínimo cambio permitido en el ciclo $epoch.")
                break
            end
        end
    end

    # Devolver el vector de pérdidas durante el entrenamiento
    return loss_history
end


function trainClassCascadeANN(maxNumNeurons::Int,
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    transferFunction::Function=σ,
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.001, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)

    #Trasponemos las matrices de entrada y de salida

    inputs = trainingDataset[1]' # el "'" se usa para transponer la matriz
    outputs = trainingDataset[2]'

    #Cambiamos el tipo de la matriz de entradas

    inputs = Float32.(inputs) 

    #Creamos una red neuronal sin capas ocultas con la función newClassCascadeNetwork

    new=newClassCascadeNetwork(length(inputs), length(outputs))

    #Entrenamos la red neuronal previa que hemos creado

    loss=trainClassANN!(new, trainingDataset, false) #devuelve un vector

    #Ahora, con un bucle, vamos a entrenar la red neuronal con una nueva capa en cada iteración

    for i in 1:maxNumNeurons

        new=addClassCascadeNeuron(new, transferFunction) #añadimos una nueva capa a la neurona

        if new.numOutputs > 1 #si la red neuronal tiene más de una neurona, entrenamos la red neuronal congelando todas las capas excepto las dos últimas
        #esto está bien????
        
            loss=vcat(loss, trainClassANN!(new, trainingDataset, trainOnly2LastLayers=true)[2]) #obviamos el primer valor del vector de loss

        end

        loss=vcat(loss, trainClassANN!(new, trainingDataset, trainOnly2LastLayers=false)[2])

    end

    return (new, loss)

end;


function trainClassCascadeANN(maxNumNeurons::Int,
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    transferFunction::Function=σ,
    maxEpochs::Int=100, minLoss::Real=0.0, learningRate::Real=0.01, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)
    
    #Convertimos el vector de salidas en una matriz

    outputs = reshape(trainingDataset[2], 1, length(trainingDataset[2])) #reshape convierte el vector en una matriz de 1 fila y tantas columnas como elementos tenga el vector

    #Llamamos a la función anterior con los mismos parámetros

    return trainClassCascadeANN(maxNumNeurons, (trainingDataset[1], outputs), transferFunction=transferFunction, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, minLossChange=minLossChange, lossChangeWindowSize=lossChangeWindowSize) 


end;
    

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

HopfieldNet = Array{Float32,2}

function trainHopfield(trainingSet::AbstractArray{<:Real,2})
    
    # Número de neuronas (dimensión de los patrones)
    num_p,num_neurons = size(trainingSet)

    # Calcular la matriz de pesos usando operaciones matriciales
    HopfieldNet = (trainingSet' * trainingSet) / num_p
    
    # Asegurar que los elementos en la diagonal son cero (sin auto-conexiones)
    for i in 1:num_neurons
        HopfieldNet[i, i] = 0.0
    end

    return Float32.(HopfieldNet)  # Devolver la matriz de pesos entrenada
end;

function trainHopfield(trainingSet::AbstractArray{<:Bool,2})

    # Convertir los valores booleanos a 1 (true) y -1 (false)
    trainingSet_real = Float32.(ifelse.(trainingSet, 1.0, -1.0))
    
    HopfieldNet = trainHopfield(trainingSet_real)

    return HopfieldNet  # Devolver la matriz de pesos entrenada
end;

function trainHopfield(trainingSetNCHW::AbstractArray{<:Bool,4})
    # Convertir el dataset de imágenes en formato NCHW a 2D
    reshaped_training_set = reshape(trainingSetNCHW,size(trainingSetNCHW,1),size(trainingSetNCHW, 2) * size(trainingSetNCHW, 3)* size(trainingSetNCHW,4))
    
    # Llamar al método correspondiente para el formato 2D
    return trainHopfield(reshaped_training_set)
end;

function stepHopfield(ann::HopfieldNet, S::AbstractArray{<:Real,1})
    
    S_float32 = Float32.(S)
    
    S_next = ann * S_float32
    
    S_next = sign.(S_next) #esta función devuelve -1 si el valor es negativo, 0 si es 0 y 1 si es positivo
    
    return S_next 

end;

function stepHopfield(ann::HopfieldNet, S::AbstractArray{<:Bool,1})
    
    S_real = Float32.(ifelse.(S, 1.0, -1.0)) #iflese devuelve 1 si el valor es true y -1 si es false
    
    S_next_real = stepHopfield(ann, S_real)
    
    S_next = S_next_real .>= 0
    
    return S_next  
end;


function runHopfield(ann::HopfieldNet, S::AbstractArray{<:Real,1})
    prev_S = nothing;
    prev_prev_S = nothing;
    while S!=prev_S && S!=prev_prev_S
        prev_prev_S = prev_S;
        prev_S = S;
        S = stepHopfield(ann, S);
    end;
    return S
end;

function runHopfield(ann::HopfieldNet, dataset::AbstractArray{<:Real,2})
    outputs = copy(dataset);
    for i in 1:size(dataset,1)
        outputs[i,:] .= runHopfield(ann, view(dataset, i, :));
    end;
    return outputs;
end;

function runHopfield(ann::HopfieldNet, datasetNCHW::AbstractArray{<:Real,4})
    outputs = runHopfield(ann, reshape(datasetNCHW, size(datasetNCHW,1), size(datasetNCHW,3)*size(datasetNCHW,4)));
    return reshape(outputs, size(datasetNCHW,1), 1, size(datasetNCHW,3), size(datasetNCHW,4));
end;

using Random

function addNoise(datasetNCHW::AbstractArray{<:Bool,4}, ratioNoise::Real)
    copia = copy(datasetNCHW)
    
    
    indices = shuffle(1:length(copia))[1:Int(round(length(copia)*ratioNoise))]

    # Invertimos los valores de los píxeles en los índices calculados
    copia[indices] .= .!copia[indices]

    println(indices)
        
    # Comentamos la siguiente línea para que los cambios en datasetNCHW no se reviertan
    # datasetNCHW = copia 
    
    return copia
end


function cropImages(datasetNCHW::AbstractArray{<:Bool,4}, ratioCrop::Real)

    copia = copy(datasetNCHW)
    
    # Obtener la dimensión de ancho (W)
    W = size(copia, 4)
    
    # Calcular el número de píxeles a mantener en el sentido horizontal
    W_keep = Int(floor(W * (1.0 - ratioCrop)))
    
    # Si ratioCrop es 0, no hay que modificar nada
    if ratioCrop == 0.0
        return datasetNCHW
    end
    
    # Crear una máscara para establecer a false los píxeles a la derecha según ratioCrop
    # La máscara tendrá valor 'false' en las posiciones a la derecha y 'true' en las demás
    # Primero, creamos un rango de índices de ancho
    indices_w = 1:W
    
    # Determinar cuáles índices deben ser establecidos a false
    mascaraw = indices_w .<= W_keep
    
    # Expandir la máscara a las dimensiones de las imágenes
    # Para esto, necesitamos agregar dimensiones de tamaño 1 en N, C y H
    # de manera que podamos realizar operaciones de broadcasting
    mascara = reshape(mascaraw, 1, 1, 1, W)
    
    # Aplicar la máscara: los píxeles a la derecha se establecerán a false
    copia .= copia .& mascara
    
    return copia
end


function randomImages(numImages::Int, resolution::Int)
    
    randomImages = randn(numImages, 1, resolution, resolution) .> 0
    return randomImages

end;

using Statistics

function averageMNISTImages(imageArray::AbstractArray{<:Real,4}, labelArray::AbstractArray{Int,1})
    
    labels = unique(labelArray)
    
    outputImages = Array{eltype(imageArray), 4}(undef, length(labels), 1, size(imageArray, 3), size(imageArray, 4))
    #eltype devuelve el tipo de los elementos de la matriz imageArray, para que la matriz outputImages tenga el mismo tipo
    #4 es el número de dimensiones de la matriz
    #undef crea una matriz de tamaño determinado con valores indefinidos
    
    for indexLabel in 1:length(labels)
        outputImages[indexLabel, 1, :, :] = dropdims(mean(imageArray[labelArray .== labels[indexLabel], 1, :, :], dims=1), dims=1)
    end
    
    return (outputImages, labels)

end;

function classifyMNISTImages(
    imageArray::AbstractArray{<:Bool,4}, 
    templateInputs::AbstractArray{<:Bool,4}, 
    templateLabels::AbstractArray{Int,1}
)
    # Crear vector de salida inicializado a -1
    outputs = fill(-1, size(imageArray, 1))

    # Iterar sobre las plantillas
    for i in 1:size(templateInputs, 1)
        template = templateInputs[i, :, :, :]  # Obtener la plantilla
        label = templateLabels[i]  # Obtener la etiqueta correspondiente

        # Compara la plantilla con todas las imágenes a clasificar
        indicesCoincidence = vec(all(imageArray .== template, dims=[3,4]))

        # Actualizar las posiciones correspondientes en el vector de salida
        outputs[indicesCoincidence] .= label
    end

    return outputs
end


function calculateMNISTAccuracies(
    datasetFolder::String, 
    labels::AbstractArray{Int,1}, 
    threshold::Real
)
    # Cargar el dataset MNIST
    trainImages, trainLabels, testImages, testLabels = loadMNIST(datasetFolder, labels, Float32)

    # Llamar a averageMNISTImages para obtener plantillas
    templateInputs, templateLabels = averageMNISTImages(trainImages, trainLabels)

    # Umbralizar las imágenes
    trainImagesBool = trainImages .>= threshold
    testImagesBool = testImages .>= threshold
    templateInputsBool = templateInputs .>= threshold

    # Entrenar la red de Hopfield
    hopfieldNet = trainHopfield(templateInputsBool)

    # Calcular precisión en el conjunto de entrenamiento
    classifiedTrainImages = stepHopfield(hopfieldNet, trainImagesBool)
    trainPredictions = classifyMNISTImages(classifiedTrainImages, templateInputsBool, templateLabels)
    trainAccuracy = mean(trainPredictions .== trainLabels)

    # Calcular precisión en el conjunto de test
    classifiedTestImages = stepHopfield(hopfieldNet, testImagesBool)
    testPredictions = classifyMNISTImages(classifiedTestImages, templateInputsBool, templateLabels)
    testAccuracy = mean(testPredictions .== testLabels)

    return trainAccuracy, testAccuracy
end


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------


#using ScikitLearn: @sk_import, fit!, predict

# #tenemos que usar SVC

# using Conda

# # # Actualizamos Conda
# Conda.update()

# # # Instalamos scikit-learn manualmente en el entorno de Conda
# Conda.add("scikit-learn")

# using ScikitLearn: @sk_import, fit!, predict
# import ScikitLearn.svm as svm
# @sk_import svm: SVC

# # Otras cosas extrañas
# ENV["PYTHON"] = ""
# Pkg.build("PyCall")  
# using Conda
# Conda.add("scikit-learn")
# using PyCall
# sk = pyimport("sklearn")

# using PyCall
# println(PyCall.python)



Batch = Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}

# lote_entero_lepiota = Tuple(loadDataset("agaricus-lepiota", "datasets"))
# hector_batch = Batch(lote_entero_lepiota)

function batchInputs(batch::Batch)
    return batch[1] # el primer elemento de la tupla
end;

function batchTargets(batch::Batch)
    return batch[2] # el segundo
end;

function batchLength(batch::Batch)
    tamanoEntradas = size(batchInputs(batch))[1]
    if  tamanoEntradas == size(batchTargets(batch))[1]
        return tamanoEntradas
    else A
        print("No miden lo mismo las entradas que las salidas, ojo con eso")
        return tamanoEntradas
    end
end;


function selectInstances(batch::Batch, indices::Any)
    return (batchInputs(batch)[indices,:], vec(batchTargets(batch)[indices,:]))
end;

#vamos a probar la función

#indices = [1,2,3,4,5,6,7,8,9,10]
#selectInstances(hector_batch, indices)

function joinBatches(batch1::Batch, batch2::Batch)
return (vcat(batchInputs(batch1), batchInputs(batch2)), vcat(batchTargets(batch1), batchTargets(batch2)))
end;


using Random, Base.Iterators, Statistics

function divideBatches(dataset::Batch, batchSize::Int; shuffleRows::Bool=false)

    if shuffleRows
        indices = shuffle(1:batchLength(dataset))
    else
        indices = 1:batchLength(dataset)
    end

    return [selectInstances(dataset, i) for i in partition(indices, batchSize)]
    
    
end;

#divideBatches(hector_batch, 10, shuffleRows=true)


function trainSVM(dataset::Batch, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.,
    supportVectors::Batch=( Array{eltype(dataset[1]),2}(undef,0,size(dataset[1],2)) , Array{eltype(dataset[2]),1}(undef,0) ) )

    # Concatenate support vectors with the training dataset
    trainingData = joinBatches(supportVectors, dataset)

    # Create the SVM model
    model = SVC(kernel=kernel, C=C, gamma=gamma, coef0=coef0, degree=degree, random_state=1)

    # Train the model
    fit!(model, batchInputs(trainingData), batchTargets(trainingData))

    # Get the indices of the new support vectors
    indicesNewSupportVectors = sort(model.support_ .+ 1)

    # Number of support vectors from the previous training
    numSupportVectors = batchLength(supportVectors)

    # Separate indices into those from the previous support vectors and the new training data
    indicesOldSupportVectors = indicesNewSupportVectors[indicesNewSupportVectors .<= numSupportVectors]
    indicesNewTrainingData = indicesNewSupportVectors[indicesNewSupportVectors .> numSupportVectors] .- numSupportVectors

    # Create the new batch of support vectors
    newSupportVectors = joinBatches(
        selectInstances(supportVectors, indicesOldSupportVectors),
        selectInstances(dataset, indicesNewTrainingData)
    )

    return model, newSupportVectors, (indicesOldSupportVectors, indicesNewTrainingData)
end;


function trainSVM(batches::AbstractArray{<:Batch,1}, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)

    # Definir un batch de vectores de soporte vacío
    supportVectors = Batch((Array{Float64, 2}(undef, 0, size(batches[1][1], 2)), Array{Float64, 1}(undef, 0)))

    # Variable para almacenar el modelo entrenado
    model = nothing

    # Iterar por todos los lotes de datos
    for batch in batches
        model, supportVectors, _ = trainSVM(batch, kernel, C, degree=degree, gamma=gamma, coef0=coef0, supportVectors=supportVectors)
    end

    # Devolver el último modelo entrenado
    return model
end;





# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------



function initializeStreamLearningData(datasetFolder::String, windowSize::Int, batchSize::Int)
    
    load = loadStreamLearningDataset(datasetFolder) 

    memoria = selectInstances(load,1:windowSize)

    resto = selectInstances(load,windowSize+1:batchLength(load))

    batches = divideBatches(resto, batchSize, shuffleRows=false)

    return (memoria,batches)
end;




## Ejemplo de cómo llamar a la función
# datasetFolder = "datasets"
# windowSize = 100  # Tamaño de la ventana inicial
# batchSize = 50    # Tamaño de los batches

# (memory,batches) = initializeStreamLearningData(datasetFolder, windowSize, batchSize)

function addBatch!(memory::Batch, newBatch::Batch)
        
        # Desplazar los datos en la memoria
        memoryInputs = vcat(batchInputs(memory)[size(newBatch[1], 1) + 1:end, :], batchInputs(newBatch))
        memoryTargets = vcat(batchTargets(memory)[size(newBatch[2], 1) + 1:end], batchTargets(newBatch))
        
        # Actualizar la memoria
        memory[1] .= memoryInputs
        memory[2] .= memoryTargets
        
        return memory
    end;
#print(addBatch!(memory, batches))


# import Pkg; Pkg.add("ScikitLearn")
using ScikitLearn: predict
using ScikitLearn: fit!

function streamLearning_SVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)

    # Inicializar memoria y batches mediante la función initializeStreamLearningData
    memoria, batches = initializeStreamLearningData(datasetFolder, windowSize, batchSize)

    # Entrenar el primer SVM mediante la función trainSVM de la práctica anterior
    svm, = trainSVM(memoria, kernel, C, degree=degree, gamma=gamma, coef0=coef0)

    # Crear un vector con tantos elementos como lotes de datos, para almacenar las precisiones
    vector_precisiones = Vector{Float64}(undef, length(batches))

    for i in eachindex(batches)
        # Hacer test del modelo actual (función predict de Scikit-Learn) con el
        # i-ésimo batch, calcular la precisión y almacenarla en el vector.
        predicciones = predict(svm, batchInputs(batches[i]))
        precision = mean(predicciones .== batchTargets(batches[i])) # necesitamos hacer la media de las predicciones porque es un vector de booleanos
        vector_precisiones[i] = precision

        # Actualizar la memoria con el i-ésimo batch mediante la función addBatch!
        addBatch!(memoria, batches[i])

        # Entrenar un nuevo SVM con la memoria actualizada que se tiene
        svm, = trainSVM(memoria, kernel, C, degree=degree, gamma=gamma, coef0=coef0)
    end

    # Finalmente, devolver el vector con las precisiones usando cada batch como test.
    return vector_precisiones
end;

#vamos a probar la función

# datasetFolder = "datasets"
# windowSize = 100
# batchSize = 50
# C = 1.0
# kernel="rbf"

# streamLearning_SVM(datasetFolder, windowSize, batchSize, kernel, C)


function streamLearning_ISVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)

    # Inicializar memoria y batches usando initializeStreamLearningData
    memory, batches = initializeStreamLearningData(datasetFolder, batchSize, batchSize)

    # Entrenar el primer SVM con la memoria inicial
    svm, supportVectors, indicesSupportVectorsInFirstBatch = trainSVM(memory, kernel, C, degree=degree, gamma=gamma, coef0=coef0)

    # Crear un vector con la edad de los patrones
    patternAges = collect(batchSize:-1:1)
    supportVectorAges = patternAges[indicesSupportVectorsInFirstBatch[1]]

    # Crear un vector para almacenar las precisiones (longitud igual a la cantidad de batches)
    precisions = Vector{Float64}(undef, length(batches))

    # Bucle para iterar sobre los batches
    for i in 1:length(batches)
        # Predecir con el modelo actual en el i-ésimo batch y calcular la precisión
        predictions = predict(svm, batchInputs(batches[i]))
        precisions[i] = mean(predictions .== batchTargets(batches[i]))

        # Actualizar el vector de edad de los vectores de soporte
        supportVectorAges .+= batchSize

        # Seleccionar vectores de soporte cuya edad es <= windowSize
        validIndices = findall(x -> x <= windowSize, supportVectorAges)
        supportVectors = selectInstances(supportVectors, validIndices)
        supportVectorAges = supportVectorAges[validIndices]

        # Entrenar un nuevo SVM con el nuevo batch y los vectores de soporte
        svm, newSupportVectors, (indicesOldSupportVectors, indicesNewTrainingData) = trainSVM(batches[i], kernel, C, degree=degree, gamma=gamma, coef0=coef0, supportVectors=supportVectors)

        # Crear un nuevo lote de datos con los nuevos vectores de soporte
        newSupportVectorsBatch = joinBatches(
            selectInstances(supportVectors, indicesOldSupportVectors),
            selectInstances(batches[i], indicesNewTrainingData)
        )

        # Crear el vector de edades de los nuevos vectores de soporte
        newSupportVectorAges = vcat(
            supportVectorAges[indicesOldSupportVectors],
            patternAges[indicesNewTrainingData]
        )

        # Actualizar los vectores de soporte y sus edades
        supportVectors = newSupportVectorsBatch
        supportVectorAges = newSupportVectorAges
    end

    # Devolver el vector de precisiones
    return precisions
end;

   

# import Pkg; Pkg.add("StatsBase")
#vamos a probar la función con un ejemplo

# datasetFolder = "datasets/"
# windowSize = 100
# batchSize = 50
# kernel = "rbf"
# C = 1.0

# streamLearning_ISVM(datasetFolder, windowSize, batchSize, kernel, C)

function euclideanDistances(memory::Batch, instance::AbstractArray{<:Real,1})
    # Calcular la distancia euclidiana entre la instancia y todas las instancias en la memoria, y cambiar el resultado al tipo AbstractArray{<:Real}
     return vec(sqrt.(sum((batchInputs(memory) .- instance').^2, dims=2)))
    
end;

#vamos a probar la función
lote_entero_lepiota = Tuple(loadDataset("agaricus-lepiota", "datasets"))
hector_batch = Batch(lote_entero_lepiota)

memory = selectInstances(hector_batch,1:10)
instance = batchInputs(hector_batch)[1,:]

euclideanDistances(memory, instance)



using StatsBase #para usar la función mode

function predictKNN(memory::Batch, instance::AbstractArray{<:Real,1}, k::Int)

    distances = euclideanDistances(memory, instance)

    min_index = partialsortperm(vec(distances), 1:k)

    return mode(memory[2][min_index])

end;

#vamos a probar la función

# memory = selectInstances(hector_batch,1:10)
# instance = batchInputs(hector_batch)[1,:]
# k = 3

#predictKNN(memory, instance, k)

function predictKNN(memory::Batch, instances::AbstractArray{<:Real,2}, k::Int)
    return [predictKNN(memory, instance, k) for instance in eachrow(instances)]
end;

#vamos a probar la función

# memory = selectInstances(hector_batch,1:10)
# instances = batchInputs(hector_batch)[1:3,:]
# k = 3

# predictKNN(memory, instances, k)


function streamLearning_KNN(datasetFolder::String, windowSize::Int, batchSize::Int, k::Int)
    
    memory, batches = initializeStreamLearningData(datasetFolder, windowSize, batchSize)
    precisions = Vector{Float64}(undef, length(batches))

    for i in 1:length(batches)
       
        predictions = predictKNN(memory, batchInputs(batches[i]), k)
        precisions[i] = mean(predictions .== batchTargets(batches[i]))

        addBatch!(memory, batches[i])
    end

    # Devolver el vector de precisiones
    return precisions

end;

#vamos a probar la función

# datasetFolder = "datasets"
# windowSize = 100
# batchSize = 50
# k = 3

# streamLearning_KNN(datasetFolder, windowSize, batchSize, k)

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 6 --------------------------------------------
# ----------------------------------------------------------------------------------------------

function predictKNN_SVM(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int, C::Real)
    
    distances = euclideanDistances(dataset, instance)
    
    minIndices = partialsortperm(vec(distances), 1:k)
    
    uniqueTargets = unique(batchTargets(selectInstances(dataset, minIndices)))
    
    if length(uniqueTargets) == 1
        return uniqueTargets[1]
    end

    model = SVC(kernel="linear", C=C, random_state=1)
    fit!(model, batchInputs(selectInstances(dataset, minIndices)), batchTargets(selectInstances(dataset, minIndices)))
    
    prediction = predict(model, reshape(instance, 1, :))[1]
    
    return prediction

end;

#vamos a probar la función

# dataset = selectInstances(hector_batch,1:10)
# instance = batchInputs(hector_batch)[1,:]
# k = 3
# C = 1.0

# predictKNN_SVM(dataset, instance, k, C)


function predictKNN_SVM(dataset::Batch, instances::AbstractArray{<:Real,2}, k::Int, C::Real)
   
    return [predictKNN_SVM(dataset, instance, k, C) for instance in eachrow(instances)]

end

#vamos a probar la función

# dataset = selectInstances(hector_batch,1:10)
# instances = batchInputs(hector_batch)[1:3,:]
# k = 3
# C = 1.0

# predictKNN_SVM(dataset, instances, k, C)



function streamLearning_SVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)

    #Inicializar memoria y batches mediante la función initializeStreamLearningData

    memoria, batches = initializeStreamLearningData(datasetFolder, windowSize, batchSize)

    #Entrenar el primer SVM mediante la función trainSVM de la práctica anterior

    svm, newsoportvector, indices = trainSVM(memoria, kernel, C, degree=degree, gamma=gamma, coef0=coef0)
    #Crear un vector con tantos elementos como lotes de datos, para almacenar las precisiones

    vector_precisiones=Vector{Float64}(undef, length(batches))

    for i in 1:length(batches)

        #Hacer test del modelo actual (función predict de Scikit-Learn) con el
        #i-ésimo batch, calcular la precisión y almacenarla en el vector.

        predicciones = predict(svm, batchInputs(batches[i]))
        precision = mean(predicciones .== batchTargets(batches[i]))
        vector_precisiones[i] = precision

        #Actualizar la memoria con el i-ésimo batch mediante la función addBatch!

        memoria = addBatch!(memoria, batches[i])

        #Entrenar un nuevo SVM con la memoria actualizada que se tiene

        svm, newsoportvector, indices = trainSVM(batches[i], kernel, C, degree=degree, gamma=gamma, coef0=coef0, supportVectors=newsoportvector)

    end

    return vector_precisiones

end;

#vamos a probar la función

# datasetFolder = "datasets"
# windowSize = 100
# batchSize = 50
# kernel = "rbf"
# C = 1.0

# streamLearning_SVM(datasetFolder, windowSize, batchSize, kernel, C)