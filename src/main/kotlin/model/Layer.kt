package model

/**
 * Layer is a layer of Nodes in the network.
 */
interface Layer<T : Node> {
    val nodes: List<T>

    val size: Int
        get() = nodes.size

    fun randomize(){
        nodes.forEach { it.randomize() }
    }

    fun clear() {
        nodes.forEach { it.clear() }
    }
}

/**
 * TransmittingLayer is a layer of TransmitterNodes in the network, i.e. any layer except the very last.
 */
interface TransmittingLayer<T : TransmitterNode> : Layer<T> {
    val dimensions: Dimensions

    fun <R : ReceiverNode> fullyConnectTo(nextLayer: ReceivingLayer<R>) {
        nodes.forEach { inputNode ->
            nextLayer.nodes.forEach { outputNode ->
                inputNode.connectTo(outputNode)
            }
        }
    }

    fun getNodeAt(x: Int, y: Int): T? {
        if(x < 0 || y < 0 || x >= dimensions.x || y >= dimensions.y){
            return null
        }

        return nodes[y * dimensions.x + x]
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
    override val dimensions: Dimensions,
    override val nodes: List<InputNode>
) : TransmittingLayer<InputNode> {

    constructor(dimensions: Dimensions) : this(
        dimensions = dimensions,
        nodes = (0 until dimensions.pixels).map { InputNode() }
    )

}

/**
 * HiddenLayer is one of the hidden intermediary layers of the network, consisting of HiddenNodes.
 */
class HiddenLayer(
    override val dimensions: Dimensions,
    override val nodes: List<HiddenNode>
) : TransmittingLayer<HiddenNode>, ReceivingLayer<HiddenNode> {

    constructor(dimensions: Dimensions, activationFunction: ActivationFunction) : this(
        dimensions = dimensions,
        nodes = (0 until dimensions.pixels).map { HiddenNode(activationFunction) }
    )

}

/**
 * OutputLayer is the very last layer of the network, consisting of OutputNodes, and from where the classification
 * result is read.
 */
class OutputLayer(
    override val nodes: List<OutputNode>
) : ReceivingLayer<OutputNode> {

    constructor(length: Int, activationFunction: ActivationFunction) : this(
        nodes = (0 until length).map { OutputNode(activationFunction) }
    )

}
