package model

import iterationMaxLength
import utils.dRelu
import utils.dSigmoid
import utils.relu
import utils.sigmoid
import kotlin.math.roundToInt
import kotlin.math.sqrt
import kotlin.random.Random

const val debugDrawing = false

/**
 * This class represents the whole network.
 */
class NeuralNetwork(
    val inputLayer: InputLayer,
    val hiddenLayers: MutableList<HiddenLayer>,
    val outputLayer: OutputLayer,
    val random: Random
) {
    companion object {
        fun build(
            inputDimensions: Dimensions,
            hiddenLayers: List<LayerParameters>,
            outputLayer: LayerParameters,
            random: Random
        ): NeuralNetwork {
            return NeuralNetwork(
                inputLayer = InputLayer(
                    nodes = (0 until inputDimensions.pixels).map {
                        InputNode(
                            activation = 0.0,
                            outputs = mutableListOf()
                        )
                    },
                    dimensions = inputDimensions
                ),
                hiddenLayers = hiddenLayers.map { layer ->
                    HiddenLayer(
                        nodes = (0 until layer.numberOfNodes).map {
                            HiddenNode(
                                activation = 0.0,
                                outputs = mutableListOf(),
                                inputs = mutableListOf(),
                                bias = 0.0,
                                activationFunction = layer.activationFunction.get(),
                                dActivationFunction = layer.activationFunction.derivative()
                            )
                        }
                    )
                }.toMutableList(),
                outputLayer = OutputLayer(
                    nodes = (0 until outputLayer.numberOfNodes).map {
                        OutputNode(
                            activation = 0.0,
                            inputs = mutableListOf(),
                            bias = 0.0,
                            activationFunction = outputLayer.activationFunction.get(),
                            dActivationFunction = outputLayer.activationFunction.derivative()
                        )
                    }
                ),
                random = random
            )
        }
    }

    fun connectAllAdjacentLayers() {
        if (hiddenLayers.isEmpty()) {
            inputLayer.fullyConnectTo(outputLayer, random)
        } else {
            inputLayer.fullyConnectTo(hiddenLayers.first(), random)
            hiddenLayers.zipWithNext().forEach { (a, b) -> a.fullyConnectTo(b, random) }
            hiddenLayers.last().fullyConnectTo(outputLayer, random)
        }
    }

    fun connectAllLayersToEveryLayer() {
        for (i in transmittingLayers.indices){
            for(j in i until receivingLayers.size) {
                transmittingLayers[i].fullyConnectTo(receivingLayers[j], random)
            }
        }
    }

    fun eval(image: List<Double>): Int {
        setInput(image)
        propagateForward()
        return getBestGuess()
    }

    fun train(
        trainingData: List<Example>,
        iterations: Int,
        batchSizeByIteration: (Int) -> Int,
        stepSizeByIteration: (Int) -> Double,
        evaluateTestData: () -> Double,
        evaluateTestDataAfterBatches: Int = 25
    ) {
        println("Samples, Average cost, Success rate over test data")

        var sample = 0L
        for (i in 0 until iterations) {
            val batchSize = batchSizeByIteration(i)
            val stepSize = stepSizeByIteration(i)

            val iterationData = trainingData
                .shuffled(random)
                .slice(0 until iterationMaxLength)

            val batches = (0 until iterationData.size / batchSize).map { i ->
                val offset = i * batchSize
                iterationData.subList(offset, offset + batchSize)
            }

            batches.forEachIndexed { batchNumber, batch ->

                allLayers.forEach { it.clear() }

                batch.forEach { example ->
                    sample++
                    inputLayer.nodes.forEachIndexed { i, node ->
                        node.activation = example.data[i]
                    }

                    hiddenLayers.forEach { layer ->
                        layer.nodes.forEach { it.derivativesOfCostByActivation.clear() }
                    }

                    outputLayer.nodes.forEachIndexed { i, node ->
                        node.desiredActivation = if (example.label == i) 1.0 else 0.0
                    }

                    propagateForward()

                    if (debugDrawing){
                        for (y in 0 until inputLayer.dimensions.y){
                            for (x in 0 until inputLayer.dimensions.x){
                                inputLayer.nodes[inputLayer.dimensions.x * y + x].activation.let {
                                    print(if(it > 0.5) "x " else if (it > 0.2) ". " else "  ")
                                }
                            }
                            println()
                        }
                        println()

                        println("Edges -")
                        printEdgeScannerLayer(hiddenLayers[0])

                        println("Edges |")
                        printEdgeScannerLayer(hiddenLayers[1])

                        println("Edges /")
                        printEdgeScannerLayer(hiddenLayers[2])

                        println("Edges \\")
                        printEdgeScannerLayer(hiddenLayers[3])
                    }

                    receivingLayers.asReversed().forEach { it.train() }

                    costs.add(outputLayer.nodes.sumByDouble { it.cost() })
                }

                if ((batchNumber + 1) % evaluateTestDataAfterBatches == 0) {
                    println("${sample}, ${costs.average()}, ${evaluateTestData()}")
                    costs.clear()
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

    private val transmittingLayers: List<TransmittingLayer<*>>
        get() = listOf(inputLayer, *hiddenLayers.toTypedArray())

    private val receivingLayers: List<ReceivingLayer<*>>
        get() = listOf(*hiddenLayers.toTypedArray(), outputLayer)

    private fun setInput(input: List<Double>) {
        if (input.size != inputLayer.size) {
            throw IllegalArgumentException("Input array must have size ${inputLayer.size}")
        }

        input.zip(inputLayer.nodes).forEach { (value, node) -> node.activation = value }
    }

    private fun propagateForward() {
        listOf(*hiddenLayers.toTypedArray(), outputLayer).forEach { layer -> layer.eval() }
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

data class LayerParameters(
    val numberOfNodes: Int,
    val activationFunction: ActivationFunction
)

enum class ActivationFunction {
    SIGMOID,
    RELU;

    fun get(): (Double) -> Double {
        return when (this) {
            SIGMOID -> { x: Double -> sigmoid(x) }
            RELU -> { x: Double -> relu(x) }
        }
    }

    fun derivative(): (Double) -> Double {
        return when (this) {
            SIGMOID -> { x: Double -> dSigmoid(x) }
            RELU -> { x: Double -> dRelu(x) }
        }
    }

}

private fun printEdgeScannerLayer(layer: HiddenLayer) {
    val dimension = sqrt(layer.size.toDouble()).roundToInt()
    for (y in 0 until dimension){
        for (x in 0 until dimension){
            layer.nodes[dimension * y + x].activation.let {
                print(if(it > 0.8) "X " else if (it > 0.3) "x "  else if (it > 0.1) ". " else if (it < 0.0) "err"  else "  ")
            }
        }
        println()
    }
    println()
}
