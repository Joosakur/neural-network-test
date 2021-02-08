import java.lang.Error
import kotlin.random.Random


interface Node {
    var activation: Double
}

interface TransmitterNode: Node {
    val outputs: MutableList<Edge>

    fun connectTo(outputNode: ReceiverNode, random: Random){
        val edge = Edge(
            inputNode = this,
            outputNode = outputNode,
            weight = random.nextDouble() - 0.5
        )

        this.outputs.add(edge)
        outputNode.inputs.add(edge)
    }
}

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
        bias -= stepSize * derivativeOfCostByBiasOnExampleK.average()
        if(bias.isNaN()) {
            throw Error("bias NaN")
        }
        inputs.forEach {
            it.weight -= stepSize * it.derivativeOfCostByWeightOnExampleK.average()
            if(it.weight.isNaN()) {
                throw Error("weight NaN")
            }
        }
    }
}

class InputNode(
    override var activation: Double,
    override val outputs: MutableList<Edge>
) : TransmitterNode

class HiddenNode(
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

            if(edge.inputNode is HiddenNode){
                edge.inputNode.derivativesOfCostByActivation.add(
                    derivativesOfCostByActivation.averageBy { dCost ->
                        c * dCost
                    }
                )
            }
        }
    }
}

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

            if(edge.inputNode is HiddenNode){
                edge.inputNode.derivativesOfCostByActivation.add(
                    edge.weight * q
                )
            }
        }
    }
}
