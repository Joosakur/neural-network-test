package networks

import model.*

class ExampleNetwork1 : NetworkBuilder {
    override fun buildNetwork(
        inputDimensions: Dimensions
    ): NeuralNetwork {
        val network = NeuralNetwork(
            inputLayer = InputLayer(inputDimensions),
            hiddenLayers = emptyList(),
            outputLayer = OutputLayer(
                length = 10,
                activationFunction = ActivationFunction.SIGMOID
            )
        )

        network.inputLayer.fullyConnectTo(network.outputLayer)

        return network
    }
}
