# La vella y querida

function trainClassCascadeANN(maxNumNeurons::Int,
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    transferFunction::Function=σ,
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.001, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)
   
    # Transponemos las matrices de entrada y salida
    X = trainingDataset[1]'
    Y = trainingDataset[2]'

    # Convertimos la matriz de entrada a Float32
    X = Float32.(X)

    # Creamos una RNA sin capas ocultas
    ann = newClassCascadeNetwork(size(X, 1), size(Y, 1))

    # Entrenamos la RNA
    trainingLosses = trainClassANN!(ann, (X, Y), false, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, minLossChange=minLossChange, lossChangeWindowSize=lossChangeWindowSize)

    # Vector de valores de loss de tipo Float32
    allTrainingLosses = Float32[]

    # Concatenamos todas las pérdidas iniciales
    allTrainingLosses = vcat(allTrainingLosses, trainingLosses)

    # Bucle con tantas iteraciones como maxNumNeurons
    for i in 1:maxNumNeurons

        # Creamos una nueva RNA con una nueva capa con una neurona
        ann = addClassCascadeNeuron(ann; transferFunction=transferFunction)

        # Si el número de capas/neuronas de esta RNA es mayor que 1, es momento de entrenar la RNA congelando todas las capas excepto las dos últimas
        if i > 1

            # Entrenamos la RNA congelando todas las capas excepto las dos últimas
            trainingLosses = trainClassANN!(ann, (X, Y), true, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, minLossChange=minLossChange, lossChangeWindowSize=lossChangeWindowSize)

            # Concatenamos las pérdidas sin omitir el primer valor
            allTrainingLosses = vcat(allTrainingLosses, trainingLosses[2:end])
        end

        # Entrenamos toda la RNA
        trainingLosses = trainClassANN!(ann, (X, Y), false, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, minLossChange=minLossChange, lossChangeWindowSize=lossChangeWindowSize)

        # Concatenamos las pérdidas sin omitir el primer valor
        allTrainingLosses = vcat(allTrainingLosses, trainingLosses[2:end])

    end

    return (ann, allTrainingLosses)
end


# La nueva y flamante
function trainClassCascadeANN(
    maxNumNeurons::Int,  # Número máximo de neuronas a añadir
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},  # Datos de entrenamiento
    transferFunction::Function = identity;  # Función de activación (por defecto identidad)
    maxEpochs::Int = 1000,  # Máximo de épocas para entrenar
    minLoss::Real = 0.0,  # Mínimo valor de pérdida para detener el entrenamiento
    learningRate::Real = 0.001,  # Tasa de aprendizaje
    minLossChange::Real = 1e-7,  # Mínimo cambio en la pérdida para detener el entrenamiento
    lossChangeWindowSize::Int = 5  # Tamaño de la ventana para monitorear el cambio en la pérdida
)
    # Extraemos entradas (X) y salidas (Y) del dataset, y las transponemos
    X, Y = trainingDataset
    X = convert(Matrix{Float32}, X')  # Convertir las entradas a Float32 y trasponer
    Y = Y'  # Trasponer las salidas

    # Crear la primera RNA sin capas ocultas, usando la función newClassCascadeNetwork
    numInputs = size(X, 1)  # Número de entradas
    numOutputs = size(Y, 1)  # Número de salidas

    ann = newClassCascadeNetwork(numInputs, numOutputs)

    # Entrenamos la primera RNA usando la función trainClassANN!
    allTrainingLosses = trainClassANN!(ann, (X, Y), false;
                                  maxEpochs=maxEpochs,
                                  minLoss=minLoss,
                                  learningRate=learningRate,
                                  minLossChange=minLossChange,
                                  lossChangeWindowSize=lossChangeWindowSize)

    # Bucle para añadir neuronas en cascada, hasta alcanzar maxNumNeurons
    for _ in 1:maxNumNeurons
        # Añadir una nueva neurona en cascada a la RNA actual
        ann = addClassCascadeNeuron(ann, transferFunction=transferFunction)

        # Si ya tenemos más de una neurona, entrenamos congelando todas las capas menos las dos últimas
        if length(ann.layers) > 1
            # Entrenar la RNA con solo las dos últimas capas ajustables
            partialLosses = trainClassANN!(ann, (X, Y), true;
                                           maxEpochs=maxEpochs,
                                           minLoss=minLoss,
                                           learningRate=learningRate,
                                           minLossChange=minLossChange,
                                           lossChangeWindowSize=lossChangeWindowSize)
            # Concatenamos las pérdidas, ignorando el primer valor
            allTrainingLosses = vcat(allTrainingLosses, partialLosses[2:end])
        end

        # Entrenar la RNA ajustando todas las capas
        trainingLosses = trainClassANN!(ann, (X, Y), false;
                                    maxEpochs=maxEpochs,
                                    minLoss=minLoss,
                                    learningRate=learningRate,
                                    minLossChange=minLossChange,
                                    lossChangeWindowSize=lossChangeWindowSize)
        # Concatenamos las pérdidas, ignorando el primer valor
        allTrainingLosses = vcat(allTrainingLosses, trainingLosses[2:end])
    end

    # Devolvemos la RNA entrenada y el vector de pérdidas acumuladas
    return (ann, allTrainingLosses)
end
