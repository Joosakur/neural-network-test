import model.*
import utils.readData
import java.io.File
import kotlin.random.Random

/**
* Main entrypoint
* */
fun main() {
    val random = Random(randomSeed)

    val trainingData = readData(
        labelsFile = File("train-labels.idx1-ubyte"),
        imagesFile = File("train-images.idx3-ubyte")
    )

    val testData = readData(
        labelsFile = File("t10k-labels.idx1-ubyte"),
        imagesFile = File("t10k-images.idx3-ubyte")
    )

    if (trainingData.dimensions != testData.dimensions) {
        throw Error("Training and test images should have same dimensions")
    }

    val network = NeuralNetwork.build(
        inputDimensions = trainingData.dimensions,
        hiddenLayers = hiddenLayers,
        outputLayer = LayerParameters(
            numberOfNodes = 10,
            activationFunction = outputLayerActivationFunction
        ),
        random = random
    )

    val horizontalEdgeScanner = createScanLayer(
        inputLayer = network.inputLayer,
        pattern = horizontalWeightPattern
    )
    network.hiddenLayers.add(horizontalEdgeScanner)

    val verticalEdgeScanner = createScanLayer(
        inputLayer = network.inputLayer,
        pattern = verticalWeightPattern
    )
    network.hiddenLayers.add(verticalEdgeScanner)

    val slashEdgeScanner = createScanLayer(
        inputLayer = network.inputLayer,
        pattern = slashWeightPattern
    )
    network.hiddenLayers.add(slashEdgeScanner)

    val backslashEdgeScanner = createScanLayer(
        inputLayer = network.inputLayer,
        pattern = backslashWeightPattern
    )
    network.hiddenLayers.add(backslashEdgeScanner)

    horizontalEdgeScanner.fullyConnectTo(network.outputLayer, random)
    verticalEdgeScanner.fullyConnectTo(network.outputLayer, random)
    slashEdgeScanner.fullyConnectTo(network.outputLayer, random)
    backslashEdgeScanner.fullyConnectTo(network.outputLayer, random)

    val test: () -> Double = {
        // the first 5000 are supposed to be easier
        val examples = testData.examples.let { if(easierTestData) it.subList(0, 5000) else it }

        var correct = 0
        var incorrect = 0
        for (example in examples) {
            val label = network.eval(example.data)
            if (label == example.label) correct++ else incorrect++
        }

        1.0 * correct / (correct + incorrect)
    }

    network.train(
        trainingData = trainingData.examples.shuffled(random),
        iterations = iterations,
        batchSizeByIteration = batchSizeByIteration,
        stepSizeByIteration = stepSizeByIteration,
        evaluateTestData = test,
        evaluateTestDataAfterBatches = evaluateTestDataAfterBatches
    )

    println("Finished! Final success rate was ${100.0 * test()} %")
}
