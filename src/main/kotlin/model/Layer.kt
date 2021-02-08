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

/**
 * InputLayer is the very first layer of the network, consisting of InputNodes, and where the input data is set.
 */
class InputLayer(
    override val nodes: List<InputNode>
) : TransmittingLayer<InputNode> {

    override fun clear() {
        for (n in nodes.indices) {
            nodes[n].outputs.forEach { edge -> edge.derivativeOfCostByWeightOnExampleK.clear() }
        }
    }

}

/**
 * HiddenLayer is one of the hidden intermediary layers of the network, consisting of HiddenNodes.
 */
class HiddenLayer(
    override val nodes: List<HiddenNode>
) : TransmittingLayer<HiddenNode>, ReceivingLayer<HiddenNode> {

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
