package model

import random

/**
 * Edge is a connection between two Nodes. It has an associated weight which portrays the connection strength.
 */
data class Edge(
    val inputNode: TransmitterNode,
    val outputNode: ReceiverNode,
    var weight: Double,
    val derivativesOfCostByWeightPerSample: MutableList<Double> = mutableListOf()
) {
    val inputActivation: Double
        get() = inputNode.activation

    val inputStrength: Double
        get() = weight * inputActivation

    fun clear() = derivativesOfCostByWeightPerSample.clear()

    fun randomize() {
        weight = random.nextDouble() - 0.5
    }
}
