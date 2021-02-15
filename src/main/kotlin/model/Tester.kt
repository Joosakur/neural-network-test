package model

import input.Sample
import output.outputNetworkActivationData

class Tester (
    val testSamples: List<Sample>,
    val outputActivationDataOnSuccess: Boolean = false,
    val outputActivationDataOnError: Boolean = false
){
    fun runTestsAndGetSuccessRate(network: NeuralNetwork): Double {
        if(outputActivationDataOnSuccess || outputActivationDataOnError){
            println("Writing data to files in the output directory")
        }

        var correct = 0
        var incorrect = 0
        testSamples.forEachIndexed { index, sample ->
            val label = network.eval(sample.data)
            if (label == sample.label) {
                correct++

                if(outputActivationDataOnSuccess) {
                    outputNetworkActivationData(
                        network,
                        "output/test/success/$index"
                    )
                }
            } else {
                incorrect++

                if(outputActivationDataOnError) {
                    outputNetworkActivationData(
                        network,
                        "output/test/error/$index"
                    )
                }
            }
        }

        return 1.0 * correct / (correct + incorrect)
    }
}
