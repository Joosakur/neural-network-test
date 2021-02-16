import input.readSamplesFromIdxFiles
import model.*
import networks.*
import output.exportToJson
import java.io.File
import kotlin.math.pow

/**
* Main entrypoint. Trains and tests the network.
* */
fun main() {
    File("output/").takeIf { it.exists() && it.isDirectory }?.deleteRecursively()

    val trainingData = readSamplesFromIdxFiles(
        labelsFile = File("train-labels.idx1-ubyte").takeIf { it.exists() } ?: File("train-labels-idx1-ubyte"),
        imagesFile = File("train-images.idx3-ubyte").takeIf { it.exists() } ?: File("train-images-idx3-ubyte")
    )

    val testData = readSamplesFromIdxFiles(
        labelsFile = File("t10k-labels.idx1-ubyte").takeIf { it.exists() } ?: File("t10k-labels-idx1-ubyte"),
        imagesFile = File("t10k-images.idx3-ubyte").takeIf { it.exists() } ?: File("t10k-images-idx3-ubyte")
    )

    if (trainingData.dimensions != testData.dimensions) {
        throw Error("Training and test images should have same dimensions")
    }

//    val network = ExampleNetwork1().buildNetwork(testData.dimensions)
//    val network = ExampleNetwork2().buildNetwork(testData.dimensions)
//    val network = ExampleNetwork3().buildNetwork(testData.dimensions)
    val network = ExampleNetwork4().buildNetwork(testData.dimensions)

    network.train(
        trainingSamples = trainingData.samples,
        tester = Tester(testData.samples),
        iterations = 12,
        iterationMaxLength = 10000,
        batchSizeByIteration = { iteration: Int -> 50 + 10 * iteration },
        stepSizeByIteration = { iteration: Int -> 1.0 * 0.97.pow(iteration) },
        evaluateTestDataAfterBatches = 25
    )

    val finalSuccessRate = Tester(
        testSamples = testData.samples,
        outputActivationDataOnSuccess = false,
        outputActivationDataOnError = false
    ).runTestsAndGetSuccessRate(network)

    println("Finished! Final success rate was ${100.0 * finalSuccessRate} %")

    File("network-export.json").writeText(exportToJson(network))
}
