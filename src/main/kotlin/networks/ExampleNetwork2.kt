package networks

import model.*

class ExampleNetwork2 : NetworkBuilder {
    override fun buildNetwork(
        inputDimensions: Dimensions
    ): NeuralNetwork {
        val network = NeuralNetwork(
            inputLayer = InputLayer(inputDimensions),
            hiddenLayers = listOf(
                HiddenLayer(
                    dimensions = Dimension1D(16),
                    activationFunction = ActivationFunction.RELU
                ),
                HiddenLayer(
                    dimensions = Dimension1D(16),
                    activationFunction = ActivationFunction.RELU
                )
            ),
            outputLayer = OutputLayer(
                length = 10,
                activationFunction = ActivationFunction.SIGMOID
            )
        )

        network.connectAllAdjacentLayers()

        return network
    }
}
