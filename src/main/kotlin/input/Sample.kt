package input

import model.Dimensions

data class Sample(
    val data: List<Double>,
    val label: Int
)

data class Samples(
    val samples: List<Sample>,
    val dimensions: Dimensions
)
