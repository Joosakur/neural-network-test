interface Node {
    var activation: Double
}

data class Edge(
    val inputNode: TransmitterNode,
    val outputNode: ReceiverNode,
    var weight: Double = 0.0
) {
    val input: Double
        get() = weight * inputNode.activation
}

interface TransmitterNode: Node {
    val outputs: MutableList<Edge>

    fun connectTo(outputNode: ReceiverNode){
        val edge = Edge(inputNode = this, outputNode = outputNode)
        this.outputs.add(edge)
        outputNode.inputs.add(edge)
    }
}

interface ReceiverNode : Node {
    val inputs: MutableList<Edge>
    val activationFunction: (n: Double) -> Double
    var bias: Double

    fun eval() {
        activation = activationFunction(inputs.sumOf(Edge::input) - bias)
    }
}

class InputNode(
    override var activation: Double,
    override val outputs: MutableList<Edge>
) : TransmitterNode

class HiddenNode(
    override var activation: Double,
    override val outputs: MutableList<Edge>,
    override val inputs: MutableList<Edge>,
    override var bias: Double,
    override val activationFunction: (n: Double) -> Double
) : TransmitterNode, ReceiverNode

open class OutputNode(
    override var activation: Double,
    override val inputs: MutableList<Edge>,
    override var bias: Double,
    override val activationFunction: (n: Double) -> Double
) : ReceiverNode

interface Layer<T : Node> {
    val nodes: List<T>

    val size: Int
        get() = nodes.size
}

interface TransmittingLayer<T : TransmitterNode> : Layer<T> {
    fun <R : ReceiverNode> fullyConnectTo(nextLayer: Layer<R>){
        nodes.forEach{ inputNode ->
            nextLayer.nodes.forEach{ outputNode ->
                inputNode.connectTo(outputNode)
            }
        }
    }
}

interface ReceivingLayer<T : ReceiverNode> : Layer<T> {
    fun eval(){
        nodes.forEach(ReceiverNode::eval)
    }
}

class InputLayer(
    override val nodes: List<InputNode>
) : TransmittingLayer<InputNode>

class HiddenLayer(
    override val nodes: List<HiddenNode>
) : TransmittingLayer<HiddenNode>, ReceivingLayer<HiddenNode>

class OutputLayer(
    override val nodes: List<OutputNode>
) : ReceivingLayer<OutputNode>

class NeuralNetwork(
    val inputLayer: InputLayer,
    val hiddenLayers: List<HiddenLayer>,
    val outputLayer: OutputLayer
){
    companion object {
        fun build(
            inputLength: Int,
            hiddenLayerLengths: List<Int>,
            outputLength: Int,
            activationFunction: (n: Double) -> Double
        ): NeuralNetwork{
            return NeuralNetwork(
                inputLayer = InputLayer(
                    nodes = (1..inputLength).map {
                        InputNode(
                            activation = 0.0,
                            outputs = mutableListOf()
                        )
                    }
                ),
                hiddenLayers = hiddenLayerLengths.map { layerLength ->
                    HiddenLayer(
                        nodes = (1..layerLength).map {
                            HiddenNode(
                                activation = 0.0,
                                outputs = mutableListOf(),
                                inputs = mutableListOf(),
                                bias = 0.0,
                                activationFunction = activationFunction
                            )
                        }
                    )
                },
                outputLayer = OutputLayer(
                    nodes = (1..outputLength).map {
                        OutputNode(
                            activation = 0.0,
                            inputs = mutableListOf(),
                            bias = 0.0,
                            activationFunction = activationFunction
                        )
                    }
                )
            )
        }
    }

    fun fullyConnect() {
        if(hiddenLayers.isEmpty()){
            inputLayer.fullyConnectTo(outputLayer)
        } else {
            inputLayer.fullyConnectTo(hiddenLayers.first())
            hiddenLayers.zipWithNext(HiddenLayer::fullyConnectTo)
            hiddenLayers.last().fullyConnectTo(outputLayer)
        }
    }

    fun eval() {
        listOf(*hiddenLayers.toTypedArray(), outputLayer).forEach{ layer -> layer.eval() }
    }

    fun setInput(input: Array<Double>){
        if(input.size != inputLayer.size){
            throw IllegalArgumentException("Input array must have size ${inputLayer.size}")
        }

        input.zip(inputLayer.nodes).forEach { (value, node) -> node.activation = value }
    }

    fun getBestGuess(): Int {
        val activations = outputLayer.nodes.map { it.activation }
        return activations.maxOrNull()
            ?.let { activations.indexOf(it) }
            ?: throw IllegalStateException("Output layers is empty")
    }
}
