import model.LayerParameters
import model.NeuralNetwork
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

    if (trainingData.pixelsPerImage != testData.pixelsPerImage) {
        throw Error("Training and test images should have same dimensions")
    }

    val network = NeuralNetwork.build(
        inputLength = trainingData.pixelsPerImage,
        hiddenLayers = hiddenLayers,
        outputLayer = LayerParameters(
            numberOfNodes = 10,
            activationFunction = outputLayerActivationFunction
        ),
        random = random
    )

    network.fullyConnect()

    val test: () -> Double = {
        var correct = 0
        var incorrect = 0
        // use the first 5000 which are supposed to be easier
        for (example in testData.examples.subList(0, 5000)) {
            val label = network.eval(example.data)
            if (label == example.label) correct++ else incorrect++
        }

        1.0 * correct / (correct + incorrect)
    }

    network.train(
        trainingData = trainingData.examples.shuffled(random).slice(0 until trainingImages),
        iterations = iterations,
        batchSize = batchSize,
        stepSize = stepSize,
        evaluateTestData = test,
        evaluateTestDataAfterBatches = evaluateTestDataAfterBatches
    )

    println("Finished! Final success rate was ${100.0 * test()} %")
}
