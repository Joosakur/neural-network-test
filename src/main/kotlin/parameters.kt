import model.ActivationFunction
import model.LayerParameters

/*
* Change these parameters as you wish.
* */

const val stepSize: Double = 0.01
const val batchSize: Int = 200
val hiddenLayers = listOf<LayerParameters>(
    LayerParameters(numberOfNodes = 32, activationFunction = ActivationFunction.RELU)
)
val outputLayerActivationFunction: ActivationFunction = ActivationFunction.SIGMOID

const val trainingImages: Int = 60000 // max 60000
const val iterations: Int = 100
const val evaluateTestDataAfterBatches: Int = 25
const val randomSeed: Long = 1234L
