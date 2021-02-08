package model

import utils.dRelu
import utils.dSigmoid
import utils.relu
import utils.sigmoid
import kotlin.random.Random

/**
 * This class represents the whole network.
 */
class NeuralNetwork(
    val inputLayer: InputLayer,
    val hiddenLayers: List<HiddenLayer>,
    val outputLayer: OutputLayer,
    val random: Random
) {
    companion object {
        fun build(
            inputLength: Int,
            hiddenLayers: List<LayerParameters>,
            outputLayer: LayerParameters,
            random: Random
        ): NeuralNetwork {
            return NeuralNetwork(
                inputLayer = InputLayer(
                    nodes = (0 until inputLength).map {
                        InputNode(
                            activation = 0.0,
                            outputs = mutableListOf()
                        )
                    }
                ),
                hiddenLayers = hiddenLayers.map { layer ->
                    HiddenLayer(
                        nodes = (0 until layer.numberOfNodes).map {
                            HiddenNode(
                                activation = 0.0,
                                outputs = mutableListOf(),
                                inputs = mutableListOf(),
                                bias = random.nextDouble() - 0.5,
                                activationFunction = layer.activationFunction.get(),
                                dActivationFunction = layer.activationFunction.derivative()
                            )
                        }
                    )
                },
                outputLayer = OutputLayer(
                    nodes = (0 until outputLayer.numberOfNodes).map {
                        OutputNode(
                            activation = 0.0,
                            inputs = mutableListOf(),
                            bias = random.nextDouble() - 0.5,
                            activationFunction = outputLayer.activationFunction.get(),
                            dActivationFunction = outputLayer.activationFunction.derivative()
                        )
                    }
                ),
                random = random
            )
        }
    }

    fun fullyConnect() {
        if (hiddenLayers.isEmpty()) {
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

    fun train(
        trainingData: List<Example>,
        iterations: Int,
        batchSize: Int,
        stepSize: Double,
        evaluateTestData: () -> Double,
        evaluateTestDataAfterBatches: Int = 25
    ) {
        println("Samples, Average cost over ${evaluateTestDataAfterBatches * batchSize} samples, Success rate over test data")

        val batches = (0 until trainingData.size / batchSize).map { i ->
            val offset = i * batchSize
            trainingData.subList(offset, offset + batchSize)
        }

        for (i in 0 until iterations) {
            batches.forEachIndexed() { batchNumber, batch ->

                allLayers.forEach { it.clear() }

                batch.forEach { example ->
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

                    receivingLayers.asReversed().forEach { it.train() }

                    costs.add(outputLayer.nodes.sumByDouble { it.cost() })
                }

                if ((batchNumber + 1) % evaluateTestDataAfterBatches == 0) {
                    println("${i * trainingData.size + (batchNumber + 1) * batchSize}, ${costs.average()}, ${evaluateTestData()}")
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
