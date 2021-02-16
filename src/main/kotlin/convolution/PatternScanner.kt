package convolution

import model.*
import kotlin.math.absoluteValue
import kotlin.math.ceil
import kotlin.math.roundToInt

fun createPatternScanLayer(
    inputLayer: InputLayer,
    pattern: Array<DoubleArray>,
    xPadding: Int = 3,
    yPadding: Int = 3,
    stepX: Int = 2,
    stepY: Int = 2
): HiddenLayer {
    val neurons = mutableListOf<StaticNeuron>()
    var x = xPadding
    var y = yPadding

    while (y < inputLayer.dimensions.y - yPadding){
        while (x < inputLayer.dimensions.x - xPadding){
            neurons.add(createMaskNeuron(inputLayer, pattern, x, y))
            x += stepX
        }
        y += stepY
        x = xPadding
    }

    return HiddenLayer(
        neurons = neurons,
        dimensions = Dimensions(
            x = ceil(1.0 * (inputLayer.dimensions.x - 2 * xPadding) / stepX).roundToInt(),
            y = ceil(1.0 * (inputLayer.dimensions.y - 2 * yPadding) / stepY).roundToInt()
        )
    )
}

val horizontalWeightPattern = arrayOf(
    doubleArrayOf(-0.4, -0.5, -0.7, -1.0, -0.7, -0.5, -0.4),
    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    doubleArrayOf(0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7),
    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    doubleArrayOf(-0.4, -0.5, -0.7, -1.0, -0.7, -0.5, -0.4)
)

val verticalWeightPattern = arrayOf(
    doubleArrayOf(-0.4, 0.0, 0.0, 0.7, 0.0, 0.0, -0.4),
    doubleArrayOf(-0.5, 0.0, 0.0, 0.8, 0.0, 0.0, -0.5),
    doubleArrayOf(-0.7, 0.0, 0.0, 0.9, 0.0, 0.0, -0.7),
    doubleArrayOf(-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0),
    doubleArrayOf(-0.7, 0.0, 0.0, 0.9, 0.0, 0.0, -0.7),
    doubleArrayOf(-0.5, 0.0, 0.0, 0.8, 0.0, 0.0, -0.5),
    doubleArrayOf(-0.4, 0.0, 0.0, 0.7, 0.0, 0.0, -0.4)
)

val slashWeightPattern = arrayOf(
    doubleArrayOf(-2.0, -1.0, -0.5, 0.0, 0.0, 0.0, 0.7),
    doubleArrayOf(-1.0, -0.5, 0.0, 0.0, 0.0, 0.8, 0.0),
    doubleArrayOf(-0.5, 0.0, 0.0, 0.7, 0.9, 0.0, 0.0),
    doubleArrayOf(0.0, 0.0, 0.7, 1.0, 0.7, 0.0, 0.0),
    doubleArrayOf(0.0, 0.0, 0.9, 0.7, 0.0, 0.0, -0.5),
    doubleArrayOf(0.0, 0.8, 0.0, 0.0, 0.0, -0.5, -1.0),
    doubleArrayOf(0.7, 0.0, 0.0, 0.0, -0.5, -1.0, -2.0)
)

val backslashWeightPattern = arrayOf(
    doubleArrayOf(0.7, 0.0, 0.0, 0.0, -0.5, -1.0, -2.0),
    doubleArrayOf(0.0, 0.8, 0.0, 0.0, 0.0, -0.5, -1.0),
    doubleArrayOf(0.0, 0.0, 0.9, 0.7, 0.0, 0.0, -0.5),
    doubleArrayOf(0.0, 0.0, 0.7, 1.0, 0.7, 0.0, 0.0),
    doubleArrayOf(-0.5, 0.0, 0.0, 0.7, 0.9, 0.0, 0.0),
    doubleArrayOf(-1.0, -0.5, 0.0, 0.0, 0.0, 0.8, 0.0),
    doubleArrayOf(-2.0, -1.0, -0.5, 0.0, 0.0, 0.0, 0.7)
)

private fun createMaskNeuron(
    inputLayer: InputLayer,
    pattern: Array<DoubleArray>,
    atX: Int,
    atY: Int
): StaticNeuron {
    val neuron = StaticNeuron(
        activationFunction = ActivationFunction.RELU
    )

    neuron.bias = -1.0

    for (i in -3..3){
        for(j in -3..3){
            val weight = pattern[j+3][i+3].takeIf { it.absoluteValue > 0.001 } ?: continue
            val inputNeuron = inputLayer.getNeuronAt(x = atX + i, y = atY + j) ?: continue
            val connection = Connection(
                inputNeuron = inputNeuron,
                outputNeuron = neuron,
                weight = weight
            )
            inputNeuron.outputs.add(connection)
            neuron.inputs.add(connection)
        }
    }

    return neuron
}
