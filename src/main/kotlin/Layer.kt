import kotlin.random.Random

interface Layer<T : Node> {
    val nodes: List<T>

    val size: Int
        get() = nodes.size

    fun clear()
}

interface TransmittingLayer<T : TransmitterNode> : Layer<T> {
    fun <R : ReceiverNode> fullyConnectTo(nextLayer: ReceivingLayer<R>, random: Random){
        nodes.forEach{ inputNode ->
            nextLayer.nodes.forEach{ outputNode ->
                inputNode.connectTo(outputNode, random)
            }
        }
    }
}

interface ReceivingLayer<T : ReceiverNode> : Layer<T> {
    fun eval(){
        nodes.forEach(ReceiverNode::eval)
    }

    fun train(){
        nodes.forEach { it.train() }
    }

    fun nudgeParameters(stepSize: Double) {
        nodes.forEach { it.nudgeParameters(stepSize) }
    }
}

class InputLayer(
    override val nodes: List<InputNode>
) : TransmittingLayer<InputNode>{

    override fun clear() {
        for (n in nodes.indices){
            nodes[n].outputs.forEach { edge -> edge.derivativeOfCostByWeightOnExampleK.clear() }
        }
    }

}

class HiddenLayer(
    override val nodes: List<HiddenNode>
) : TransmittingLayer<HiddenNode>, ReceivingLayer<HiddenNode>{

    override fun clear() {
        nodes.forEach { node ->
            node.derivativeOfCostByBiasOnExampleK.clear()
            node.outputs.forEach { edge -> edge.derivativeOfCostByWeightOnExampleK.clear() }
        }
    }

}

class OutputLayer(
    override val nodes: List<OutputNode>
) : ReceivingLayer<OutputNode>{

    override fun clear() {
        for (n in nodes.indices){
            nodes[n].derivativeOfCostByBiasOnExampleK.clear()
        }
    }
}
