package model

import kotlin.random.Random

/**
 * Layer is a layer of Nodes in the network.
 */
interface Layer<T : Node> {
    val nodes: List<T>

    val size: Int
        get() = nodes.size

    fun clear()
}

/**
 * TransmittingLayer is a layer of TransmitterNodes in the network, i.e. any layer except the very last.
 */
interface TransmittingLayer<T : TransmitterNode> : Layer<T> {
    fun <R : ReceiverNode> fullyConnectTo(nextLayer: ReceivingLayer<R>, random: Random) {
        nodes.forEach { inputNode ->
            nextLayer.nodes.forEach { outputNode ->
                inputNode.connectTo(outputNode, random)
            }
        }
    }
}

/**
 * ReceivingLayer is a layer of ReceiverNodes in the network, i.e. any layer except the very first.
 */
interface ReceivingLayer<T : ReceiverNode> : Layer<T> {
    fun eval() {
        nodes.forEach(ReceiverNode::eval)
    }

    fun train() {
        nodes.forEach { it.train() }
    }

    fun nudgeParameters(stepSize: Double) {
        nodes.forEach { it.nudgeParameters(stepSize) }
    }
}

data class Dimensions (
    val x: Int,
    val y: Int
){
    val pixels: Int
        get() = x * y
}

/**
 * InputLayer is the very first layer of the network, consisting of InputNodes, and where the input data is set.
 */
class InputLayer(
    override val nodes: List<InputNode>,
    val dimensions: Dimensions
) : TransmittingLayer<InputNode> {

    override fun clear() {
        for (n in nodes.indices) {
            nodes[n].outputs.forEach { edge -> edge.derivativeOfCostByWeightOnExampleK.clear() }
        }
    }

    fun getNodeAt(x: Int, y: Int): InputNode? {
        if(x < 0 || y < 0 || x >= dimensions.x || y >= dimensions.y){
            return null
        }
        return nodes[y * dimensions.x + x]
    }

}

/**
 * HiddenLayer is one of the hidden intermediary layers of the network, consisting of HiddenNodes.
 */
class HiddenLayer(
    override val nodes: List<HiddenNode>
) : TransmittingLayer<HiddenNode>, ReceivingLayer<HiddenNode> {
    companion object {
        fun build(
            params: LayerParameters,
            random: Random
        ) = HiddenLayer(
            nodes = (0 until params.numberOfNodes).map {
                HiddenNode(
                    activation = 0.0,
                    outputs = mutableListOf(),
                    inputs = mutableListOf(),
                    bias = random.nextDouble() - 0.5,
                    activationFunction = params.activationFunction.get(),
                    dActivationFunction = params.activationFunction.derivative()
                )
            }
        )
    }

    override fun clear() {
        nodes.forEach { node ->
            node.derivativeOfCostByBiasOnExampleK.clear()
            node.outputs.forEach { edge -> edge.derivativeOfCostByWeightOnExampleK.clear() }
        }
    }

}

/**
 * OutputLayer is the very last layer of the network, consisting of OutputNodes, and from where the classification
 * result is read.
 */
class OutputLayer(
    override val nodes: List<OutputNode>
) : ReceivingLayer<OutputNode> {

    override fun clear() {
        for (n in nodes.indices) {
            nodes[n].derivativeOfCostByBiasOnExampleK.clear()
        }
    }
}
