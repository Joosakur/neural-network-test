package output

import model.NeuralNetwork
import model.TransmittingLayer
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import kotlin.math.roundToInt

fun outputNetworkActivationData(
    network: NeuralNetwork,
    path: String
) {
    saveLayerActivationAsImage(
        network.inputLayer,
        path,
        "input"
    )

    network.hiddenLayers.forEachIndexed { i, layer ->
        saveLayerActivationAsImage(
            layer,
            path,
            "hidden-layer-$i"
        )
    }

    File("$path/output.txt").printWriter().use { printWriter ->
        val result = network.getBestGuess()
        val confidence = 100.0 *
            network.outputLayer.neurons[result].activation /
            network.outputLayer.neurons.sumByDouble { it.activation }

        printWriter.println("Result: $result")
        printWriter.println("Confidence: $confidence %")
        printWriter.println()
        network.outputLayer.neurons.forEachIndexed { index, outputNeuron ->
            printWriter.println("$index: ${outputNeuron.activation}")
        }
    }
}

private fun saveLayerActivationAsImage(
    layer: TransmittingLayer<*>,
    path: String,
    name: String
){
    val width = layer.dimensions.x
    val height = layer.dimensions.y

    val pixels = layer.neurons.map { it.activation }
        .let { activations ->
            val min = activations.minOfOrNull { it }
            if (min != null && min < 0) {
                activations.map { it - min }
            } else activations
        }
        .let { activations ->
            val max = activations.maxOrNull()
            if (max != null && max > 1) {
                activations.map { it / max }
            } else activations
        }
        .map { (it * 255).roundToInt() }

    val image = BufferedImage(
        width,
        height,
        BufferedImage.TYPE_INT_RGB
    )
    val raster = image.raster
    for (y in 0 until height){
        for(x in 0 until width){
            val value = pixels[y * width + x]
            raster.setPixel(x, y, intArrayOf(value, value, value))
        }
    }

    if(!File("$path/").exists()) {
        File("$path/").mkdirs()
    }

    ImageIO.write(
        image,
        "png",
        File("$path/$name.png")
    )
}
