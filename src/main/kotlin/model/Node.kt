package model

import random
import utils.*
import java.util.*

/**
 * Node presents a single node (or a neuron) in the network.
 */
interface Node {
    val id: UUID
    var activation: Double

    fun clear()

    fun randomize()
}

/**
 * TransmitterNode is a Node which has output connections to other nodes, i.e. it can be any node except one on the
 * very last layer.
 */
interface TransmitterNode : Node {
    val outputs: MutableList<Edge>

    fun connectTo(outputNode: ReceiverNode) {
        val edge = Edge(
            inputNode = this,
            outputNode = outputNode,
            weight = random.nextDouble() - 0.5
        )

        this.outputs.add(edge)
        outputNode.inputs.add(edge)
    }

    override fun clear(){
        outputs.forEach { it.clear() }
    }

    override fun randomize() {
        outputs.forEach { it.randomize() }
    }
}

/**
 * ReceiverNode is a Node which has input connections from other nodes, i.e. it can be any node except one on the
 * very last layer.
 */
interface ReceiverNode : Node {
    val inputs: MutableList<Edge>
    var bias: Double
    val derivativesOfCostByBiasPerSample: MutableList<Double>
    val activationFunction: ActivationFunction

    fun z(): Double = inputs.sumByDouble { it.inputStrength } + bias

    fun eval() {
        activation = activationFunction.eval(z())
    }

    fun train()

    fun nudgeParameters(stepSize: Double) {
        bias = (bias - stepSize * derivativesOfCostByBiasPerSample.average()).coerceIn(-0.8, 0.8)
        if (bias.isNaN()) {
            throw Error("bias NaN")
        }
        inputs.forEach {
            it.weight -= stepSize * it.derivativesOfCostByWeightPerSample.average()
            if (it.weight.isNaN()) {
                throw Error("weight NaN")
            }
        }
    }

    override fun clear(){
        derivativesOfCostByBiasPerSample.clear()
    }

    override fun randomize() {
        bias = random.nextDouble() - 0.5
    }
}

/**
 * InputNode is a Node on the input layer.
 */
class InputNode : TransmitterNode{
    override val id: UUID = UUID.randomUUID()
    override var activation: Double = 0.0
    override val outputs: MutableList<Edge> = mutableListOf()
}

/**
 * HiddenNode is a Node on one of the hidden intermediary layers.
 */
open class HiddenNode(
    override val activationFunction: ActivationFunction,
) : TransmitterNode, ReceiverNode {
    override val id: UUID = UUID.randomUUID()
    override var activation: Double = 0.0
    override var bias: Double = 0.0
    override val inputs: MutableList<Edge> = mutableListOf()
    override val outputs: MutableList<Edge> = mutableListOf()
    override val derivativesOfCostByBiasPerSample: MutableList<Double> = mutableListOf()
    val derivativesOfCostByActivation: MutableList<Double> = mutableListOf()

    override fun train() {
        val a = activationFunction.evalDerivative(z())

        derivativesOfCostByBiasPerSample.add(
            derivativesOfCostByActivation.averageBy { dcda ->
                a * dcda
            }
        )

        inputs.forEach { edge ->
            val b = edge.inputActivation * a
            val c = edge.weight * a

            edge.derivativesOfCostByWeightPerSample.add(
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

    override fun clear() {
        super<TransmitterNode>.clear()
        super<ReceiverNode>.clear()
    }

    override fun randomize() {
        super<TransmitterNode>.randomize()
        super<ReceiverNode>.randomize()
    }
}

/**
 * StaticNode is a preconfigured HiddenNode that does not learn on its own.
 */
class StaticNode(
    activationFunction: ActivationFunction
) : HiddenNode(
    activationFunction = activationFunction
) {
    override val id: UUID = UUID.randomUUID()

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
    override val activationFunction: ActivationFunction,
) : ReceiverNode {
    override val id: UUID = UUID.randomUUID()
    override var activation: Double = 0.0
    override var bias: Double = 0.0
    override val inputs: MutableList<Edge> = mutableListOf()
    override val derivativesOfCostByBiasPerSample: MutableList<Double> = mutableListOf()
    var desiredActivation: Double = 0.0

    fun cost() = (activation - desiredActivation).squared()

    fun dCost() = 2 * (activation - desiredActivation)

    override fun train() {
        val q = activationFunction.evalDerivative(z()) * dCost()

        derivativesOfCostByBiasPerSample.add(q)

        inputs.forEach { edge ->
            edge.derivativesOfCostByWeightPerSample.add(
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
