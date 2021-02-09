package model

import kotlin.math.absoluteValue

val horizontalWeightPattern = arrayOf(
    doubleArrayOf(-0.4, -0.5, -0.7, -1.0, -0.7, -0.5, -0.4),
    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    doubleArrayOf(0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7),
    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    doubleArrayOf(-0.4, -0.5, -0.7, -1.0, -0.7, -0.5, -0.4)
)

val verticalWeightPattern = arrayOf(
    doubleArrayOf(-0.4, 0.0, 0.0, 0.7, 0.0, 0.0, -0.4),
    doubleArrayOf(-0.5, 0.0, 0.0, 0.8, 0.0, 0.0, -0.5),
    doubleArrayOf(-0.7, 0.0, 0.0, 0.9, 0.0, 0.0, -0.7),
    doubleArrayOf(-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0),
    doubleArrayOf(-0.7, 0.0, 0.0, 0.9, 0.0, 0.0, -0.7),
    doubleArrayOf(-0.5, 0.0, 0.0, 0.8, 0.0, 0.0, -0.5),
    doubleArrayOf(-0.4, 0.0, 0.0, 0.7, 0.0, 0.0, -0.4)
)

val slashWeightPattern = arrayOf(
    doubleArrayOf(-2.0, -1.0, -0.5, 0.0, 0.0, 0.0, 0.7),
    doubleArrayOf(-1.0, -0.5, 0.0, 0.0, 0.0, 0.8, 0.0),
    doubleArrayOf(-0.5, 0.0, 0.0, 0.7, 0.9, 0.0, 0.0),
    doubleArrayOf(0.0, 0.0, 0.7, 1.0, 0.7, 0.0, 0.0),
    doubleArrayOf(0.0, 0.0, 0.9, 0.7, 0.0, 0.0, -0.5),
    doubleArrayOf(0.0, 0.8, 0.0, 0.0, 0.0, -0.5, -1.0),
    doubleArrayOf(0.7, 0.0, 0.0, 0.0, -0.5, -1.0, -2.0)
)

val backslashWeightPattern = arrayOf(
    doubleArrayOf(0.7, 0.0, 0.0, 0.0, -0.5, -1.0, -2.0),
    doubleArrayOf(0.0, 0.8, 0.0, 0.0, 0.0, -0.5, -1.0),
    doubleArrayOf(0.0, 0.0, 0.9, 0.7, 0.0, 0.0, -0.5),
    doubleArrayOf(0.0, 0.0, 0.7, 1.0, 0.7, 0.0, 0.0),
    doubleArrayOf(-0.5, 0.0, 0.0, 0.7, 0.9, 0.0, 0.0),
    doubleArrayOf(-1.0, -0.5, 0.0, 0.0, 0.0, 0.8, 0.0),
    doubleArrayOf(-2.0, -1.0, -0.5, 0.0, 0.0, 0.0, 0.7)
)

fun createMaskNode(
    inputLayer: InputLayer,
    pattern: Array<DoubleArray>,
    atX: Int,
    atY: Int
): StaticNode {
    val node = StaticNode(
        activation = 0.0,
        outputs = mutableListOf(),
        inputs = mutableListOf(),
        bias = -1.0,
        activationFunction = ActivationFunction.RELU.get()
    )
    for (i in -3..3){
        for(j in -3..3){
            val weight = pattern[j+3][i+3].takeIf { it.absoluteValue > 0.001 } ?: continue
            val inputNode = inputLayer.getNodeAt(x = atX + i, y = atY + j) ?: continue
            val edge = Edge(
                inputNode = inputNode,
                outputNode = node,
                weight = weight
            )
            inputNode.outputs.add(edge)
            node.inputs.add(edge)
        }
    }
    return node
}

fun createScanLayer(
    inputLayer: InputLayer,
    pattern: Array<DoubleArray>,
    xPadding: Int = 3,
    yPadding: Int = 3,
    stepX: Int = 2,
    stepY: Int = 2
): HiddenLayer {
    val nodes = mutableListOf<StaticNode>()
    var x = xPadding
    var y = yPadding
    while (y < inputLayer.dimensions.y - yPadding){
        while (x < inputLayer.dimensions.x - xPadding){
            nodes.add(createMaskNode(inputLayer, pattern, x, y))
            x += stepX
        }
        y += stepY
        x = xPadding
    }

    return HiddenLayer(nodes)
}
