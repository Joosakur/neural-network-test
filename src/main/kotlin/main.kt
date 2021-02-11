import input.readSamplesFromIdxFiles
import model.*
import networks.*
import java.io.File
import kotlin.math.pow

/**
* Main entrypoint
* */
fun main() {
    // clear output directory
    File("output/").takeIf { it.exists() && it.isDirectory }?.deleteRecursively()

    val trainingSamples = readSamplesFromIdxFiles(
        labelsFile = File("train-labels.idx1-ubyte"),
        imagesFile = File("train-images.idx3-ubyte")
    )

    val testSamples = readSamplesFromIdxFiles(
        labelsFile = File("t10k-labels.idx1-ubyte"),
        imagesFile = File("t10k-images.idx3-ubyte")
    )

    if (trainingSamples.dimensions != testSamples.dimensions) {
        throw Error("Training and test images should have same dimensions")
    }

//    val network = ExampleNetwork1().buildNetwork(testSamples.dimensions)
//    val network = ExampleNetwork2().buildNetwork(testSamples.dimensions)
//    val network = ExampleNetwork3().buildNetwork(testSamples.dimensions)
    val network = ExampleNetwork4().buildNetwork(testSamples.dimensions)

    network.train(
        trainingSamples = trainingSamples.samples,
        tester = Tester(testSamples),
        iterations = 10,
        iterationMaxLength = 10000,
        batchSizeByIteration = { iteration: Int -> 50 + 10 * iteration },
        stepSizeByIteration = { iteration: Int -> 1.0 * 0.97.pow(iteration) },
        evaluateTestDataAfterBatches = 25
    )

    val finalSuccessRate = Tester(
        testData = testSamples,
        outputDataOnSuccess = false,
        outputDataOnError = true
    ).runTestsAndGetSuccessRate(network)

    println("Finished! Final success rate was ${100.0 * finalSuccessRate} %")
}
