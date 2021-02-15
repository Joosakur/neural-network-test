package model

import input.Sample
import random

/**
 * This class represents the whole network.
 */
class NeuralNetwork(
    val inputLayer: InputLayer,
    val hiddenLayers: List<HiddenLayer>,
    val outputLayer: OutputLayer
) {
    fun connectAllAdjacentLayers() {
        if (hiddenLayers.isEmpty()) {
            inputLayer.fullyConnectTo(outputLayer)
        } else {
            inputLayer.fullyConnectTo(hiddenLayers.first())
            hiddenLayers.zipWithNext().forEach { (a, b) -> a.fullyConnectTo(b) }
            hiddenLayers.last().fullyConnectTo(outputLayer)
        }
    }

    fun connectAllLayersWithEachOther() {
        for (i in transmittingLayers.indices){
            for(j in i until receivingLayers.size) {
                transmittingLayers[i].fullyConnectTo(receivingLayers[j])
            }
        }
    }

    fun eval(image: List<Double>): Int {
        setInput(image)
        propagateForward()
        return getBestGuess()
    }

    fun setInput(input: List<Double>) {
        if (input.size != inputLayer.size) {
            throw IllegalArgumentException("Input array must have size ${inputLayer.size}")
        }

        input.zip(inputLayer.nodes).forEach { (value, node) -> node.activation = value }
    }

    fun propagateForward() {
        listOf(*hiddenLayers.toTypedArray(), outputLayer).forEach { layer -> layer.eval() }
    }

    fun getBestGuess(): Int {
        val activations = outputLayer.nodes.map { it.activation }
        return activations.maxOrNull()
            ?.let { activations.indexOf(it) }
            ?: throw IllegalStateException("Output layers is empty")
    }

    fun train(
        trainingSamples: List<Sample>,
        iterations: Int,
        iterationMaxLength: Int,
        batchSizeByIteration: (Int) -> Int,
        stepSizeByIteration: (Int) -> Double,
        tester: Tester,
        evaluateTestDataAfterBatches: Int = 25
    ) {
        println("Samples, Average cost, Success rate over test data")

        var sampleNumber = 0L
        for (i in 0 until iterations) {
            val batchSize = batchSizeByIteration(i)
            val stepSize = stepSizeByIteration(i)

            val iterationData = trainingSamples
                .shuffled(random)
                .slice(0 until iterationMaxLength)

            val batches = (0 until iterationData.size / batchSize).map { i ->
                val offset = i * batchSize
                iterationData.subList(offset, offset + batchSize)
            }

            batches.forEachIndexed { batchNumber, batch ->

                allLayers.forEach { it.clear() }

                batch.forEach { sample ->
                    sampleNumber++

                    inputLayer.nodes.forEachIndexed { i, node ->
                        node.activation = sample.data[i]
                    }

                    hiddenLayers.forEach { layer ->
                        layer.nodes.forEach { it.derivativesOfCostByActivation.clear() }
                    }

                    outputLayer.nodes.forEachIndexed { i, node ->
                        node.desiredActivation = if (sample.label == i) 1.0 else 0.0
                    }

                    propagateForward()

                    receivingLayers.asReversed().forEach { it.train() }

                    costs.add(outputLayer.nodes.sumByDouble { it.cost() })
                }

                if ((batchNumber + 1) % evaluateTestDataAfterBatches == 0) {
                    val successRate = tester.runTestsAndGetSuccessRate(this)
                    println("$sampleNumber, ${costs.average()}, $successRate")
                    costs.clear()
                }

                receivingLayers.forEach { it.nudgeParameters(stepSize) }
            }
        }
    }

    private val costs: MutableList<Double> = mutableListOf()

    val allLayers: List<Layer<*>>
        get() = listOf(
            inputLayer,
            *hiddenLayers.toTypedArray(),
            outputLayer
        )

    val transmittingLayers: List<TransmittingLayer<*>>
        get() = listOf(inputLayer, *hiddenLayers.toTypedArray())

    val receivingLayers: List<ReceivingLayer<*>>
        get() = listOf(*hiddenLayers.toTypedArray(), outputLayer)

}
