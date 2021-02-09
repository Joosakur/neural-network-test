import model.ActivationFunction
import model.LayerParameters
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt

/*
* Change these parameters as you wish.
* */

val stepSizeByIteration: (Int) -> Double = {
    iteration: Int -> 1.0 * 0.97.pow(iteration)
}

val batchSizeByIteration: (Int) -> Int = {
    iteration: Int -> (50 * (1 + sqrt(1.0 * iteration))).roundToInt()
}

val hiddenLayers = listOf<LayerParameters>(
)

val outputLayerActivationFunction: ActivationFunction = ActivationFunction.SIGMOID

const val iterations: Int = 100
const val iterationMaxLength: Int = 10000
const val evaluateTestDataAfterBatches: Int = 25
const val randomSeed: Long = 1234L
const val easierTestData: Boolean = false
