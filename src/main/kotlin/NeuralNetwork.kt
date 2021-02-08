import kotlin.random.Random

class NeuralNetwork(
    val inputLayer: InputLayer,
    val hiddenLayers: List<HiddenLayer>,
    val outputLayer: OutputLayer,
    val random: Random

){
    companion object {
        fun build(
            inputLength: Int,
            hiddenLayerLengths: List<Int>,
            outputLength: Int,
            activationFunction: ActivationFunction,
            random: Random
        ): NeuralNetwork{
            val f = when(activationFunction){
                ActivationFunction.SIGMOID -> {
                    x: Double -> sigmoid(x)
                }
                ActivationFunction.RELU -> {
                    x: Double -> relu(x)
                }
            }
            val df = when(activationFunction){
                ActivationFunction.SIGMOID -> {
                        x: Double -> dSigmoid(x)
                }
                ActivationFunction.RELU -> {
                        x: Double -> dRelu(x)
                }
            }

            return NeuralNetwork(
                inputLayer = InputLayer(
                    nodes = (0 until inputLength).map {
                        InputNode(
                            activation = 0.0,
                            outputs = mutableListOf()
                        )
                    }
                ),
                hiddenLayers = hiddenLayerLengths.map { layerLength ->
                    HiddenLayer(
                        nodes = (0 until layerLength).map {
                            HiddenNode(
                                activation = 0.0,
                                outputs = mutableListOf(),
                                inputs = mutableListOf(),
                                bias = random.nextDouble() - 0.5,
                                activationFunction = f,
                                dActivationFunction = df
                            )
                        }
                    )
                },
                outputLayer = OutputLayer(
                    nodes = (0 until outputLength).map {
                        OutputNode(
                            activation = 0.0,
                            inputs = mutableListOf(),
                            bias = random.nextDouble() - 0.5,
                            activationFunction = f,
                            dActivationFunction = df
                        )
                    }
                ),
                random = random
            )
        }
    }

    fun fullyConnect() {
        if(hiddenLayers.isEmpty()){
            inputLayer.fullyConnectTo(outputLayer, random)
        } else {
            inputLayer.fullyConnectTo(hiddenLayers.first(), random)
            hiddenLayers.zipWithNext().forEach { (a, b) -> a.fullyConnectTo(b, random) }
            hiddenLayers.last().fullyConnectTo(outputLayer, random)
        }
    }

    fun eval(image: List<Double>): Int {
        setInput(image)
        propagateForward()
        return getBestGuess()
    }

    fun train(batch: List<Example>, iterations: Int, miniBatchSize: Int, stepSize: Double, tester: (() -> Double)? = null){
        val miniBatches = (0 until batch.size / miniBatchSize).map { i ->
            val offset = i * miniBatchSize
            batch.subList(offset, offset + miniBatchSize)
        }

        for (i in 0 until iterations) {
            miniBatches.forEachIndexed() { batchNumber, miniBatch ->

                allLayers.forEach { it.clear() }

                miniBatch.forEach { example ->
                    inputLayer.nodes.forEachIndexed { i, node ->
                        node.activation = example.data[i]
                    }

                    hiddenLayers.forEach { layer ->
                        layer.nodes.forEach { it.derivativesOfCostByActivation.clear() }
                    }

                    outputLayer.nodes.forEachIndexed { i, node ->
                        node.desiredActivation = if(example.label == i) 1.0 else 0.0
                    }

                    propagateForward()

                    receivingLayers.asReversed().forEach{ it.train() }

                    if(tester != null) costs.add(outputLayer.nodes.sumByDouble { it.cost() })
                }

                if(tester != null) {
                    if(batchNumber % 25 == 0) {
                        println("${costs.average()}, ${tester()}")
                        costs.clear()
                    }
                } else {
                    println("Training... iteration ${i+1}/$iterations batch ${batchNumber+1}/${miniBatches.size} (${100.0 * (i * miniBatches.size + batchNumber) / (iterations * miniBatches.size)} %)")
                }

                receivingLayers.forEach { it.nudgeParameters(stepSize) }
            }
        }
    }

    private val costs: MutableList<Double> = mutableListOf()

    private val allLayers: List<Layer<*>>
        get() = listOf(
            inputLayer,
            *hiddenLayers.toTypedArray(),
            outputLayer
        )

    private val receivingLayers: List<ReceivingLayer<*>>
        get() = listOf(*hiddenLayers.toTypedArray(), outputLayer)

    private fun setInput(input: List<Double>){
        if(input.size != inputLayer.size){
            throw IllegalArgumentException("Input array must have size ${inputLayer.size}")
        }

        input.zip(inputLayer.nodes).forEach { (value, node) -> node.activation = value }
    }

    private fun propagateForward() {
        listOf(*hiddenLayers.toTypedArray(), outputLayer).forEach{ layer -> layer.eval() }
    }

    private fun getBestGuess(): Int {
        val activations = outputLayer.nodes.map { it.activation }
        return activations.maxOrNull()
            ?.let { activations.indexOf(it) }
            ?: throw IllegalStateException("Output layers is empty")
    }

}

data class Example(
    val data: List<Double>,
    val label: Int
)

enum class ActivationFunction{
    SIGMOID,
    RELU
}
