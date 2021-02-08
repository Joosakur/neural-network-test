package utils

import kotlin.math.pow

fun Double.squared() = this * this

fun <T> Collection<T>.averageBy(f: (element: T) -> Double) = this.sumByDouble(f) / this.size

fun sigmoid(x: Double) = 1 / (1 + Math.E.pow(-x))

fun dSigmoid(x: Double) = sigmoid(x) * (1 - sigmoid(x))

fun relu(x: Double) = if (x < 0.0) 0.0 else x

fun dRelu(x: Double) = if (x < 0.0) 0.0 else 1.0
