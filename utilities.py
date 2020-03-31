# # # This acts as a library for essential methods
import numpy as np
from PIL import Image


# # # Neural Network Utilities


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidInverse(x):
    return np.log(x / (1 - x))


def sigmoidDerivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def outputCostDerivative(layer, trueValue):
    # This calculates the error of output layer
    # 1. Loop through the last layer
    # 2. Calculate error of each node (error being derivative of cost)

    # Initialisation
    value = []

    # Stage One
    for i in range(len(trueValue)):

        # Stage Two
        newCostDerivative = 2 * (trueValue[i] - layer[i].output) * sigmoidDerivative(layer[i].z)
        value.append(newCostDerivative)
    return value


def evaluateCost(answer, trueValue):
    # Gets the cost of an answer
    # 1. Gets the cost of one node and sum up the total cost

    # Initialisation
    value = 0

    for i in range(len(answer)):
        # Stage One
        value += (answer[i] - trueValue[i]) ** 2
    return value


# # # Image manipulation


def loadImageFromPath(path):
    size = 28, 28
    im = Image.open(path).convert('LA')
    im.thumbnail(size)
    return im


def loadImageFromPIL(img):
    size = 28, 28
    img = img.convert('LA')
    img.thumbnail(size)
    return img


def centreImage(img):
    # This draws a box around the number and then crops out the number and puts it in the middle of another image
    # 1. Loop through image
    # 2. Get the leftmost, rightmost, bottom and topmost pixel of the image
    # 3. Remove possibility for errors with a simple check
    # 4. Crop out image and paste the result into a new images centre

    # Initialisation
    width, height = img.size
    left, right, top, bottom = 100000000000, 0, 0, 0
    tgtImg = Image.new('LA', (width, height))

    # Stage One
    for y in range(height):
        for x in range(width):
            tgtImg.putpixel((x, y), (255, 255))

            # Stage Two
            white, black = img.getpixel((x, y))
            if white != 255 or black != 255:
                if x < left:
                    left = x
                if x > right:
                    right = x
                if top == 0:
                    top = y
                bottom = y

    # Stage Three
    if left == 100000000000:
        return img

    # Stage Four
    img = img.crop((left, top, right, bottom))
    tgtImg.paste(img, (int((width / 2) - (right - left) / 2), int((height / 2) - (bottom - top) / 2)))

    return tgtImg


def cleanImage(image):
    # Changes the image into an array
    # 1. Loop though the image
    # 2. Changes the representation of the colour into a single number between 0 and 1
    # 3. Change array into numpy array

    # Initialisation
    width, height = image.size
    array = []
    adjust = 0.99 / 255

    # Stage One
    for y in range(height):
        for x in range(width):
            white, black = image.getpixel((x, y))

            # Stage Two
            if white == 255 and black == 255:
                array.append(0)
                continue
            elif white == 0 and black == 255:
                array.append(1)
                continue
            elif white > black:
                array.append((white - black) * adjust)
                continue
            else:
                array.append((black - white) * adjust)
                continue

    # 3.
    array = np.array(array)

    return array


def cropOutNumber(startX, startY, img):
    # This function crops out an image from its furthest left and highest pixel
    # 1. Outline Image
    # 2. Fix up image by filling in holes left by the outlining

    # Initialisation
    width, height = img.size
    numberFound = False
    numberPixels = []
    direction = 1
    tgtImg = Image.new('LA', (width, height))
    for x in range(width):
        for y in range(height):
            tgtImg.putpixel((x, y), (255, 255))

    x = startX
    y = startY

    # Stage One
    while not numberFound:

        numberPixels.append([x, y])

        tgtImg.putpixel((x, y), (img.getpixel((x, y))))

        if x == 0:
            x += 1
        if y == 0:
            y += 1
        if x == width - 1:
            x = width - 2
        if y == height - 1:
            y = height - 2

        if direction == 1:
            # Direction 6 ----------------------------------------------------------------------------------------------
            if [x - 1, y - 1] not in numberPixels and img.getpixel((x - 1, y - 1)) != (255, 255):
                x -= 1
                y -= 1
                direction = 2
            # Direction 7 ----------------------------------------------------------------------------------------------
            elif [x - 1, y] not in numberPixels and img.getpixel((x - 1, y)) != (255, 255):
                x -= 1
                direction = 1
            # Direction 8 ----------------------------------------------------------------------------------------------
            elif [x - 1, y + 1] not in numberPixels and img.getpixel((x - 1, y + 1)) != (255, 255):
                x -= 1
                y += 1
                direction = 1
            # Direction 1 ----------------------------------------------------------------------------------------------
            elif [x, y + 1] not in numberPixels and img.getpixel((x, y + 1)) != (255, 255):
                y += 1
                direction = 1
            # Direction 2 ----------------------------------------------------------------------------------------------
            elif [x + 1, y + 1] not in numberPixels and img.getpixel((x + 1, y + 1)) != (255, 255):
                x += 1
                y += 1
                direction = 1
            # Direction 3 ----------------------------------------------------------------------------------------------
            elif [x + 1, y] not in numberPixels and img.getpixel((x + 1, y)) != (255, 255):
                x += 1
                direction = 1
            # Direction 4 ----------------------------------------------------------------------------------------------
            elif [x + 1, y - 1] not in numberPixels and img.getpixel((x + 1, y - 1)) != (255, 255):
                x += 1
                y -= 1
                direction = 2
            # Direction 5 ----------------------------------------------------------------------------------------------
            elif [x, y - 1] not in numberPixels and img.getpixel((x, y - 1)) != (255, 255):
                y -= 1
                direction = 2
            else:
                numberFound = True

        elif direction == 2:
            # Direction 2 ----------------------------------------------------------------------------------------------
            if [x + 1, y + 1] not in numberPixels and img.getpixel((x + 1, y + 1)) != (255, 255):
                x += 1
                y += 1
                direction = 1
            # Direction 3 ----------------------------------------------------------------------------------------------
            elif [x + 1, y] not in numberPixels and img.getpixel((x + 1, y)) != (255, 255):
                x += 1
                direction = 2
            # Direction 4 ----------------------------------------------------------------------------------------------
            elif [x + 1, y - 1] not in numberPixels and img.getpixel((x + 1, y - 1)) != (255, 255):
                x += 1
                y -= 1
                direction = 2
            # Direction 5 ----------------------------------------------------------------------------------------------
            elif [x, y - 1] not in numberPixels and img.getpixel((x, y - 1)) != (255, 255):
                y -= 1
                direction = 2
            # Direction 6 ----------------------------------------------------------------------------------------------
            elif [x - 1, y - 1] not in numberPixels and img.getpixel((x - 1, y - 1)) != (255, 255):
                x -= 1
                y -= 1
                direction = 2
            # Direction 7 ----------------------------------------------------------------------------------------------
            elif [x - 1, y] not in numberPixels and img.getpixel((x - 1, y)) != (255, 255):
                x -= 1
                direction = 2
            # Direction 8 ----------------------------------------------------------------------------------------------
            elif [x - 1, y + 1] not in numberPixels and img.getpixel((x - 1, y + 1)) != (255, 255):
                x -= 1
                y += 1
                direction = 1
            # Direction 1 ----------------------------------------------------------------------------------------------
            elif [x, y + 1] not in numberPixels and img.getpixel((x, y + 1)) != (255, 255):
                y += 1
                direction = 1
            else:
                numberFound = True

    tgtImg.putpixel((x, y), img.getpixel((x, y)))

    # Stage 2
    tgtImg = fillHolesInImages(tgtImg)

    return tgtImg


def cropOutNumbers(img):
    # This isolates each number/shape from an image
    # 1. Checks whether image is blank
    # 2. Finds the furthest left and highest up pixel of a number
    # 3. Crops out the image based on the pixel found
    # 4. Removes the original number from the image
    # 5. Checks for finish and if not repeat

    # Initialisation
    width, height = img.size
    numbers = []
    startX, startY = 0, 0
    index = -1
    allNumbersFound = False

    # Stage 1
    countOne = 0
    countTwo = 0
    for x in range(width):
        for y in range(height):
            countOne += 1
            if img.getpixel((x, y)) == (255, 255):
                countTwo += 1
    if countOne == countTwo:
        allNumbersFound = True

    while not allNumbersFound:

        # Stage 2
        pixelFound = False
        for x in range(width):
            for y in range(height):
                if img.getpixel((x, y)) != (255, 255) and not pixelFound:
                    startX = x
                    startY = y
                    pixelFound = True

        # Stage 3
        numbers.append(cropOutNumber(startX, startY, img))
        index += 1

        # Stage 4
        for x in range(width):
            for y in range(height):
                if numbers[index].getpixel((x, y)) != (255, 255):
                    img.putpixel((x, y), (255, 255))

        # Stage 5
        countOne = 0
        countTwo = 0
        for x in range(width):
            for y in range(height):
                countOne += 1
                if img.getpixel((x, y)) == (255, 255):
                    countTwo += 1
        if countOne == countTwo:
            allNumbersFound = True

    return numbers


def fillHolesInImages(img):
    # This function fills up holes in images
    # 1. Invert image colours
    # 2. Flood fill outside of image with white
    # 3. Combine images
    width, height = img.size
    invImg = Image.new('LA', (width, height))

    # Inverts the colours of the input Image

    for x in range(width):
        for y in range(height):

            if img.getpixel((x, y)) != (255, 255):
                invImg.putpixel((x, y), (255, 255))
            else:
                invImg.putpixel((x, y), (0, 255))

    # Removes the outer layer of black from the image

    outImg = floodFill(0, 0, (255, 255), invImg)

    # Combines the two images

    for x in range(width):
        for y in range(width):
            if img.getpixel((x, y)) == (255, 255):
                img.putpixel((x, y), outImg.getpixel((x, y)))

    return img


def floodFill(x, y, colour, img):
    # This is a recursive function that replaces all adjacent pixels of the same colour with a different colour
    # 1. Check whether the pixel given is the right colour
    # 2. Replace it with a pixel of the right colour
    # 3. Choose the next pixel to change colour

    width, height = img.size

    # Stage One
    if img.getpixel((x, y)) == (0, 255):

        # Stage Two
        img.putpixel((x, y), colour)

        # Stage Three
        if x > 0:
            floodFill(x - 1, y, colour, img)
        if x < width - 1:
            floodFill(x + 1, y, colour, img)
        if y > 0:
            floodFill(x, y - 1, colour, img)
        if y < height - 1:
            floodFill(x, y + 1, colour, img)

    return img
