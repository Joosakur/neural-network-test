package model

/**
 * Edge is a connection between two Nodes. It has an associated weight which portrays the connection strength.
 */
data class Edge(
    val inputNode: TransmitterNode,
    val outputNode: ReceiverNode,
    var weight: Double,
    val derivativeOfCostByWeightOnExampleK: MutableList<Double> = mutableListOf()
) {
    val inputActivation: Double
        get() = inputNode.activation

    val inputStrength: Double
        get() = weight * inputActivation
}
