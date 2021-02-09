package model

import utils.averageBy
import utils.squared
import kotlin.random.Random

/**
 * Node presents a single node (or a neuron) in the network.
 */
interface Node {
    var activation: Double
}

/**
 * TransmitterNode is a Node which has output connections to other nodes, i.e. it can be any node except one on the
 * very last layer.
 */
interface TransmitterNode : Node {
    val outputs: MutableList<Edge>

    fun connectTo(outputNode: ReceiverNode, random: Random) {
        val edge = Edge(
            inputNode = this,
            outputNode = outputNode,
            weight = random.nextDouble() - 0.5
        )

        this.outputs.add(edge)
        outputNode.inputs.add(edge)
    }
}

/**
 * ReceiverNode is a Node which has input connections from other nodes, i.e. it can be any node except one on the
 * very last layer.
 */
interface ReceiverNode : Node {
    val inputs: MutableList<Edge>
    var bias: Double
    val derivativeOfCostByBiasOnExampleK: MutableList<Double>
    val activationFunction: (x: Double) -> Double
    val dActivationFunction: (x: Double) -> Double

    fun z(): Double = inputs.sumByDouble { it.inputStrength } + bias

    fun eval() {
        activation = activationFunction(z())
    }

    fun train()

    fun nudgeParameters(stepSize: Double) {
        bias = (bias - stepSize * derivativeOfCostByBiasOnExampleK.average()).coerceIn(-0.8, 0.8)
        if (bias.isNaN()) {
            throw Error("bias NaN")
        }
        inputs.forEach {
            it.weight -= stepSize * it.derivativeOfCostByWeightOnExampleK.average()
            if (it.weight.isNaN()) {
                throw Error("weight NaN")
            }
        }
    }
}

/**
 * InputNode is a Node on the input layer.
 */
class InputNode(
    override var activation: Double,
    override val outputs: MutableList<Edge>
) : TransmitterNode

/**
 * HiddenNode is a Node on one of the hidden intermediary layers.
 */
open class HiddenNode(
    override var activation: Double,
    override val outputs: MutableList<Edge>,
    override val inputs: MutableList<Edge>,
    override var bias: Double,
    override val activationFunction: (x: Double) -> Double,
    override val dActivationFunction: (x: Double) -> Double,
    override val derivativeOfCostByBiasOnExampleK: MutableList<Double> = mutableListOf(),
    val derivativesOfCostByActivation: MutableList<Double> = mutableListOf()
) : TransmitterNode, ReceiverNode {

    override fun train() {
        val a = dActivationFunction(z())

        derivativeOfCostByBiasOnExampleK.add(
            derivativesOfCostByActivation.averageBy { dcda ->
                a * dcda
            }
        )

        inputs.forEach { edge ->
            val b = edge.inputActivation * a
            val c = edge.weight * a

            edge.derivativeOfCostByWeightOnExampleK.add(
                derivativesOfCostByActivation.averageBy { dCost ->
                    b * dCost
                }
            )

            if (edge.inputNode is HiddenNode) {
                edge.inputNode.derivativesOfCostByActivation.add(
                    derivativesOfCostByActivation.averageBy { dCost ->
                        c * dCost
                    }
                )
            }
        }
    }
}

/**
 * StaticNode is a preconfigured HiddenNode that does not learn on its own.
 */
class StaticNode(
    override var activation: Double,
    override val outputs: MutableList<Edge>,
    override val inputs: MutableList<Edge>,
    override var bias: Double = 0.0,
    override val activationFunction: (x: Double) -> Double
) : HiddenNode(
    activation = activation,
    outputs = outputs,
    inputs = inputs,
    bias = bias,
    activationFunction = activationFunction,
    dActivationFunction = { _: Double -> 0.0 },
    derivativeOfCostByBiasOnExampleK = mutableListOf(),
    derivativesOfCostByActivation = mutableListOf()
) {

    override fun train() {
        // do nothing
    }

    override fun nudgeParameters(stepSize: Double) {
        // do nothing
    }
}

/**
 * OutputNode is a Node on the output layer.
 */
open class OutputNode(
    override var activation: Double,
    override val inputs: MutableList<Edge>,
    override var bias: Double,
    override val activationFunction: (x: Double) -> Double,
    override val dActivationFunction: (x: Double) -> Double,
    override val derivativeOfCostByBiasOnExampleK: MutableList<Double> = mutableListOf(),
    var desiredActivation: Double = 0.0
) : ReceiverNode {
    fun cost() = (activation - desiredActivation).squared()

    fun dCost() = 2 * (activation - desiredActivation)

    override fun train() {
        val q = dActivationFunction(z()) * dCost()

        derivativeOfCostByBiasOnExampleK.add(q)

        inputs.forEach { edge ->
            edge.derivativeOfCostByWeightOnExampleK.add(
                edge.inputActivation * q
            )

            if (edge.inputNode is HiddenNode) {
                edge.inputNode.derivativesOfCostByActivation.add(
                    edge.weight * q
                )
            }
        }
    }
}
