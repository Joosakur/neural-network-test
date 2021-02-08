import java.io.File
import kotlin.random.Random

val random = Random(1234L)

val hiddenLayerNodes = listOf<Int>()
const val iterations = 100
const val stepSize = 0.015
const val trainingImages = 60000 // max 60000
const val miniBatchSize = 200
val activationFunction: ActivationFunction = ActivationFunction.SIGMOID

val trainingData = readData(
    labelsFile = File("train-labels.idx1-ubyte"),
    imagesFile = File("train-images.idx3-ubyte")
)

val testData = readData(
    labelsFile = File("t10k-labels.idx1-ubyte"),
    imagesFile = File("t10k-images.idx3-ubyte")
)

fun main(){
    if(trainingData.pixelsPerImage != testData.pixelsPerImage) {
        throw Error("Training and test images should have same dimensions")
    }

    val network = NeuralNetwork.build(
        inputLength = trainingData.pixelsPerImage,
        hiddenLayerLengths = hiddenLayerNodes,
        outputLength = 10,
        activationFunction = activationFunction,
        random = random
    )

    network.fullyConnect()

    val test: () -> Double = {
        var correct = 0
        var incorrect = 0
        // use the first 5000 which are supposed to be easier
        for (example in testData.examples.subList(0, 5000)) {
            val label = network.eval(example.data)
            if(label == example.label) correct++ else incorrect++
        }

        1.0 * correct / (correct + incorrect)
    }

    network.train(
        batch = trainingData.examples.shuffled(random).slice(0 until trainingImages),
        iterations = iterations,
        miniBatchSize = miniBatchSize,
        stepSize = stepSize,
        tester = test
    )

    println("Finished! Final success rate was ${100.0 * test()} %")

}
