package input

import model.Dimensions
import java.io.File
import java.nio.ByteBuffer

fun readSamplesFromIdxFiles(imagesFile: File, labelsFile: File): Samples {
    val imageSet = readImages(imagesFile)
    val labels = readLabels(labelsFile)

    if (imageSet.images.size != labels.size) {
        throw Error("${labels.size} labels does not match ${imageSet.images.size} images")
    }

    return Samples(
        samples = imageSet.images.zip(labels).map { (image, label) -> Sample(data = image, label = label) },
        dimensions = imageSet.dimensions
    )
}

private fun readLabels(file: File): List<Int> {
    val bytes = file.readBytes()

    val magicNumber = read32BitInt(bytes, 0)
    if (magicNumber != 2049) throw Error("Invalid file type")

    val numberOfItems = read32BitInt(bytes, 4)

    val labels = bytes.slice(8 until bytes.size).map { it.toInt() }
    if (labels.size != numberOfItems) {
        throw Error("Expected $numberOfItems labels, found ${labels.size}")
    }

    return labels
}

private data class ImageSet (
    val images: List<List<Double>>,
    val dimensions: Dimensions
)

private fun readImages(file: File): ImageSet {
    val bytes = file.readBytes()

    val magicNumber = read32BitInt(bytes, 0)
    if (magicNumber != 2051) throw Error("Invalid file type")

    val numberOfImages = read32BitInt(bytes, 4)

    val yResolution = read32BitInt(bytes, 8)
    val xResolution = read32BitInt(bytes, 12)
    val pixelsPerImage = xResolution * yResolution

    if ((bytes.size - 16) != numberOfImages * pixelsPerImage) {
        throw Error("Number of pixels does not match")
    }

    val images = (0 until numberOfImages).map { n ->
        val offset = n * pixelsPerImage + 16
        bytes.slice(offset until offset + pixelsPerImage).map { pixel -> (pixel.toInt() and 0xFF) / 255.0 }
    }

    return ImageSet(images, Dimensions(x = xResolution, y = yResolution))
}

private fun read32BitInt(byteArray: ByteArray, offset: Int) = ByteBuffer
    .wrap(byteArray.sliceArray(offset until offset + 4))
    .int
