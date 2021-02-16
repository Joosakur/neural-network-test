package model

/**
 * Layer is a set of Neurons in the network.
 */
interface Layer<T : Neuron> {
    val neurons: List<T>

    val size: Int
        get() = neurons.size

    fun randomize(){
        neurons.forEach { it.randomize() }
    }

    fun clear() {
        neurons.forEach { it.clear() }
    }
}

/**
 * TransmittingLayer is a Layer of TransmitterNeurons, i.e. any layer except the very last.
 */
interface TransmittingLayer<T : TransmitterNeuron> : Layer<T> {
    val dimensions: Dimensions

    fun <R : ReceiverNeuron> fullyConnectTo(nextLayer: ReceivingLayer<R>) {
        neurons.forEach { inputNeuron ->
            nextLayer.neurons.forEach { outputNeuron ->
                inputNeuron.connectTo(outputNeuron)
            }
        }
    }

    fun getNeuronAt(x: Int, y: Int): T? {
        if(x < 0 || y < 0 || x >= dimensions.x || y >= dimensions.y){
            return null
        }

        return neurons[y * dimensions.x + x]
    }

}

/**
 * ReceivingLayer is a Layer of ReceiverNeurons, i.e. any layer except the very first.
 */
interface ReceivingLayer<T : ReceiverNeuron> : Layer<T> {

    fun eval() {
        neurons.forEach(ReceiverNeuron::eval)
    }

    fun train() {
        neurons.forEach { it.train() }
    }

    fun nudgeParameters(stepSize: Double) {
        neurons.forEach { it.nudgeParameters(stepSize) }
    }

}

/**
 * InputLayer is the very first layer of the network, consisting of InputNeurons, and where the input data is set.
 */
class InputLayer(
    override val dimensions: Dimensions,
    override val neurons: List<InputNeuron>
) : TransmittingLayer<InputNeuron> {

    constructor(dimensions: Dimensions) : this(
        dimensions = dimensions,
        neurons = (0 until dimensions.pixels).map { InputNeuron() }
    )

}

/**
 * HiddenLayer is one of the hidden intermediary layers of the network, consisting of HiddenNeurons.
 */
class HiddenLayer(
    override val dimensions: Dimensions,
    override val neurons: List<HiddenNeuron>
) : TransmittingLayer<HiddenNeuron>, ReceivingLayer<HiddenNeuron> {

    constructor(dimensions: Dimensions, activationFunction: ActivationFunction) : this(
        dimensions = dimensions,
        neurons = (0 until dimensions.pixels).map { HiddenNeuron(activationFunction) }
    )

}

/**
 * OutputLayer is the very last layer of the network, consisting of OutputNeurons, and from where the classification
 * result is read.
 */
class OutputLayer(
    override val neurons: List<OutputNeuron>
) : ReceivingLayer<OutputNeuron> {

    constructor(length: Int, activationFunction: ActivationFunction) : this(
        neurons = (0 until length).map { OutputNeuron(activationFunction) }
    )

}
