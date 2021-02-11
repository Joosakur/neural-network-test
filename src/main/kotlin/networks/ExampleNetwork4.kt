package networks

import convolution.*
import model.*

class ExampleNetwork4 : NetworkBuilder {
    override fun buildNetwork(
        inputDimensions: Dimensions
    ): NeuralNetwork {
        val inputLayer = InputLayer(inputDimensions)

        val hiddenLayers = listOf(
            createPatternScanLayer(
                inputLayer = inputLayer,
                pattern = horizontalWeightPattern
            ),
            createPatternScanLayer(
                inputLayer = inputLayer,
                pattern = verticalWeightPattern
            ),
            createPatternScanLayer(
                inputLayer = inputLayer,
                pattern = slashWeightPattern
            ),
            createPatternScanLayer(
                inputLayer = inputLayer,
                pattern = backslashWeightPattern
            )
        )

        val outputLayer = OutputLayer(
            length = 10,
            activationFunction = ActivationFunction.SIGMOID
        )

        hiddenLayers.forEach { it.fullyConnectTo(outputLayer) }

        return NeuralNetwork(
            inputLayer,
            hiddenLayers,
            outputLayer
        )
    }
}
