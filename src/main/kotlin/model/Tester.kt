package model

import input.Samples
import output.outputNetworkActivationData

class Tester (
    val testData: Samples,
    val outputDataOnSuccess: Boolean = false,
    val outputDataOnError: Boolean = false
){
    fun runTestsAndGetSuccessRate(network: NeuralNetwork): Double {
        if(outputDataOnSuccess || outputDataOnError){
            println("Writing data to files in the output directory")
        }

        var correct = 0
        var incorrect = 0
        testData.samples.forEachIndexed { index, sample ->
            val label = network.eval(sample.data)
            if (label == sample.label) {
                correct++

                if(outputDataOnSuccess) {
                    outputNetworkActivationData(
                        network,
                        "output/test/success/$index"
                    )
                }
            } else {
                incorrect++

                if(outputDataOnError) {
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
