package model

import random

/**
 * This is a connection between two Neurons. It has an associated weight which portrays the connection strength.
 */
data class Connection(
    val inputNeuron: TransmitterNeuron,
    val outputNeuron: ReceiverNeuron,
    var weight: Double,
    val derivativesOfCostByWeightPerSample: MutableList<Double> = mutableListOf()
) {
    val inputActivation: Double
        get() = inputNeuron.activation

    val inputStrength: Double
        get() = weight * inputActivation

    fun clear() = derivativesOfCostByWeightPerSample.clear()

    fun randomize() {
        weight = random.nextDouble() - 0.5
    }
}
