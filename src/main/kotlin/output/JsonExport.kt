package output

import model.NeuralNetwork
import model.Neuron

data class NeuronPosition(
    val layerIndex: Int,
    val neuronIndex: Int
)

private fun getNeuronPosition(network: NeuralNetwork, target: Neuron): NeuronPosition {
    network.allLayers.forEachIndexed { layerIndex, layer ->
        layer.neurons.forEachIndexed { neuronIndex, neuron ->
            if(target.id == neuron.id) {
                return NeuronPosition(
                    layerIndex = layerIndex,
                    neuronIndex = neuronIndex
                )
            }
        }
    }

    throw Error("Neuron not found")
}

fun exportToJson(network: NeuralNetwork): String {
    return """{
    "inputLayerLength": ${network.inputLayer.neurons.size},
    "otherLayers": ${network.receivingLayers.map { layer -> """
        {
            "neurons": ${layer.neurons.map { neuron -> """
                {
                    "bias": ${neuron.bias},
                    "activationFunction": "${neuron.activationFunction.name}",
                    "inputs": ${neuron.inputs.map { input -> """
                        {${getNeuronPosition(network, input.inputNeuron).let { """
                            "fromLayer": ${it.layerIndex},
                            "fromNeuron": ${it.neuronIndex},""" }}
                            "weight": ${input.weight}
                        }""" }}
                    
                }""" }}
            
        }""" }}
    
}""".replace(" ", "").replace("\n", "")
}
