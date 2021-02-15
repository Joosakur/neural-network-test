package output

import model.NeuralNetwork
import model.Node

data class NodePosition(
    val layerIndex: Int,
    val nodeIndex: Int
)

private fun getNodePosition(network: NeuralNetwork, target: Node): NodePosition {
    network.allLayers.forEachIndexed { layerIndex, layer ->
        layer.nodes.forEachIndexed { nodeIndex, node ->
            if(target.id == node.id) {
                return NodePosition(
                    layerIndex = layerIndex,
                    nodeIndex = nodeIndex
                )
            }
        }
    }

    throw Error("Node not found")
}

fun exportToJson(network: NeuralNetwork): String {
    return """{
    "inputLayerLength": ${network.inputLayer.nodes.size},
    "otherLayers": ${network.receivingLayers.map { layer -> """
        {
            "nodes": ${layer.nodes.map { node -> """
                {
                    "bias": ${node.bias},
                    "activationFunction": "${node.activationFunction.name}",
                    "inputs": ${node.inputs.map { input -> """
                        {${getNodePosition(network, input.inputNode).let { """
                            "fromLayer": ${it.layerIndex},
                            "fromNode": ${it.nodeIndex},""" }}
                            "weight": ${input.weight}
                        }""" }}
                    
                }""" }}
            
        }""" }}
    
}""".replace(" ", "").replace("\n", "")
}
