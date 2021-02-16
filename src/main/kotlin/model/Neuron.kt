package model

import random
import utils.*
import java.util.*

/**
 * Neuron presents a single neuron (or a node) in the network.
 */
interface Neuron {
    val id: UUID
    var activation: Double

    fun clear()

    fun randomize()
}

/**
 * TransmitterNeuron is a Neuron which has output connections to other neurons, i.e. it can be any neuron except one on the
 * very last layer.
 */
interface TransmitterNeuron : Neuron {
    val outputs: MutableList<Connection>

    fun connectTo(outputNeuron: ReceiverNeuron) {
        val connection = Connection(
            inputNeuron = this,
            outputNeuron = outputNeuron,
            weight = random.nextDouble() - 0.5
        )

        this.outputs.add(connection)
        outputNeuron.inputs.add(connection)
    }

    override fun clear(){
        outputs.forEach { it.clear() }
    }

    override fun randomize() {
        outputs.forEach { it.randomize() }
    }
}

/**
 * ReceiverNeuron is a Neuron which has input connections from other neurons, i.e. it can be any neuron except one on the
 * very first layer.
 */
interface ReceiverNeuron : Neuron {
    val inputs: MutableList<Connection>
    var bias: Double
    val derivativesOfCostByBiasPerSample: MutableList<Double>
    val activationFunction: ActivationFunction

    fun propagationFunction(): Double = inputs.sumByDouble { it.inputStrength } + bias

    fun eval() {
        activation = activationFunction.eval(propagationFunction())
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
 * InputNeuron is a Neuron on the input layer.
 */
class InputNeuron : TransmitterNeuron{
    override val id: UUID = UUID.randomUUID()
    override var activation: Double = 0.0
    override val outputs: MutableList<Connection> = mutableListOf()
}

/**
 * HiddenNeuron is a Neuron on one of the hidden intermediary layers.
 */
open class HiddenNeuron(
    override val activationFunction: ActivationFunction,
) : TransmitterNeuron, ReceiverNeuron {
    override val id: UUID = UUID.randomUUID()
    override var activation: Double = 0.0
    override var bias: Double = 0.0
    override val inputs: MutableList<Connection> = mutableListOf()
    override val outputs: MutableList<Connection> = mutableListOf()
    override val derivativesOfCostByBiasPerSample: MutableList<Double> = mutableListOf()
    val derivativesOfCostByActivation: MutableList<Double> = mutableListOf()

    override fun train() {
        val a = activationFunction.evalDerivative(propagationFunction())

        derivativesOfCostByBiasPerSample.add(
            derivativesOfCostByActivation.averageBy { dcda ->
                a * dcda
            }
        )

        inputs.forEach { connection ->
            val b = connection.inputActivation * a
            val c = connection.weight * a

            connection.derivativesOfCostByWeightPerSample.add(
                derivativesOfCostByActivation.averageBy { dCost ->
                    b * dCost
                }
            )

            if (connection.inputNeuron is HiddenNeuron) {
                connection.inputNeuron.derivativesOfCostByActivation.add(
                    derivativesOfCostByActivation.averageBy { dCost ->
                        c * dCost
                    }
                )
            }
        }
    }

    override fun clear() {
        super<TransmitterNeuron>.clear()
        super<ReceiverNeuron>.clear()
    }

    override fun randomize() {
        super<TransmitterNeuron>.randomize()
        super<ReceiverNeuron>.randomize()
    }
}

/**
 * StaticNeuron is a preconfigured HiddenNeuron that does not learn on its own.
 */
class StaticNeuron(
    activationFunction: ActivationFunction
) : HiddenNeuron(
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
 * OutputNeuron is a Neuron on the output layer.
 */
open class OutputNeuron(
    override val activationFunction: ActivationFunction,
) : ReceiverNeuron {
    override val id: UUID = UUID.randomUUID()
    override var activation: Double = 0.0
    override var bias: Double = 0.0
    override val inputs: MutableList<Connection> = mutableListOf()
    override val derivativesOfCostByBiasPerSample: MutableList<Double> = mutableListOf()
    var desiredActivation: Double = 0.0

    fun cost() = (activation - desiredActivation).squared()

    fun dCost() = 2 * (activation - desiredActivation)

    override fun train() {
        val q = activationFunction.evalDerivative(propagationFunction()) * dCost()

        derivativesOfCostByBiasPerSample.add(q)

        inputs.forEach { connection ->
            connection.derivativesOfCostByWeightPerSample.add(
                connection.inputActivation * q
            )

            if (connection.inputNeuron is HiddenNeuron) {
                connection.inputNeuron.derivativesOfCostByActivation.add(
                    connection.weight * q
                )
            }
        }
    }
}
