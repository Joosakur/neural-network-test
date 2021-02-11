package model

open class Dimensions (
    val x: Int,
    val y: Int
){
    val pixels: Int
        get() = x * y

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as Dimensions

        if (x != other.x) return false
        if (y != other.y) return false

        return true
    }

    override fun hashCode(): Int {
        var result = x
        result = 31 * result + y
        return result
    }

}

class Dimension1D(x: Int): Dimensions(x = x, y = 1)
