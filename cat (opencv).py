import cv2
import numpy as np
               
            # read image

img = cv2.imread("cat.jpg")

            # type of image

print(type(img))                                  # to find the image type

            # shape of image

image_shape = img.shape                           # finding shape of image
print("Image shape:", image_shape)
height, width , channel = image_shape
print("Height:", height)
print("Width:", width)
print("channel",channel)

             # rotate the image

rotationMatrix = cv2.getRotationMatrix2D((width/2, height/2), 90, .5)
rotatedImage = cv2.warpAffine(img, rotationMatrix, (width, height))

                     # cropping image

startRow = int(height*.15)
startCol = int(width*.15)
endRow = int(height*.85)
endCol = int(width*.85)
croppedImage = img[startRow:endRow, startCol:endCol]

                    # resize image

newImg = cv2.resize(img, (550, 350))


                     # contrast image

contrast_img = cv2.addWeighted(img, 2.5, np.zeros(img.shape, img.dtype), 0, 0)


                     #  gaussian blur filter image 

blur_image = cv2.GaussianBlur(img, (7,7), 0)

                    # median blur filter image

median_blur_image = cv2.medianBlur(img,5)

                    # edge image by canny filter

edge_img = cv2.Canny(img,100,200)

                    #black n white image by gray scale filter

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                     # denoised image

denoise_img = cv2.fastNlMeansDenoisingColored(img,None,20,10,7,21)

                     # showing output

cv2.imshow('denoised image',denoise_img)                 # to show denoised image
cv2.imshow('black n white image',gray_img)              # to show black n white image
cv2.imshow('edge image',edge_img)                      # edge image
cv2.imshow('median blur', median_blur_image)          # to show 50% blured image
cv2.imshow('blured image',blur_image)                # to show blur image
cv2.imshow('contrast image',contrast_img)            # to show contrast image
cv2.imshow('Rotated Image', rotatedImage)            # to show rotated image
cv2.imshow('Resized Image', newImg)                  # to resize image
cv2.imshow('Cropped Image', croppedImage)            # to show cropped image
cv2.imshow('Original Image', img)                     # to show the image
cv2.waitKey(0)