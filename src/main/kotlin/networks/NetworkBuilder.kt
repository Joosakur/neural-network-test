package networks

import model.Dimensions
import model.NeuralNetwork

interface NetworkBuilder {
    fun buildNetwork(
        inputDimensions: Dimensions
    ): NeuralNetwork
}
