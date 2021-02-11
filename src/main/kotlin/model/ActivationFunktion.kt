package model

import utils.derivativeOfRelu
import utils.derivativeOfSigmoid
import utils.relu
import utils.sigmoid

enum class ActivationFunction {
    SIGMOID,
    RELU;

    fun eval(x: Double): Double {
        return when (this) {
            SIGMOID -> sigmoid(x)
            RELU -> relu(x)
        }
    }

    fun evalDerivative(x: Double): Double {
        return when (this) {
            SIGMOID -> derivativeOfSigmoid(x)
            RELU -> derivativeOfRelu(x)
        }
    }
}
