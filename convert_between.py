Convert between Python tuple and list：

a = (1, 2)      # a is a tuple
b = list(a)     # b is a list
c = tuple(b)  # c is a tuple


Convert between Python tuple, list and NumPy 1D array

a = (1, 2)          # a is a tuple
b = np.array(a)  # b is an numpy array
c = tuple(b)       # c is a tuple

a = [1, 2]          # a is a python array
b = np.array(a)  # b is a numpy array
c = list(b)          # c is a python list


Convert between NumPy 2D array and NumPy matrix

a = numpy.ndarray([2,3])   # create 2x3 array
m1 = numpy.asmatrix(a)   # does not create new matrix, m1 refers to the same memory as a
m2 = numpy.matrix(a)      # creates new matrix and copies content

b1 = numpy.asarray(m2)   # does not create array, b1 refers to the same emory as m2
b2 = numpy.array(m2)      # creates new array and copies content



Read/Write Images with OpenCV

image = cv.LoadImage(“ponzo.jpg”)
cv.SaveImage(“out.png”, image)



Read/Write Images with PIL

image = Image.open(“ponzo.jpg”)
image.save(“out.jpg”)



Read/Write Images with PyOpenCV

mat = pyopencv.imread(“ponzo.jpg”)
pyopencv.imwrite(“out.png”, mat)



Convert between OpenCV image and PIL image

# color image
cimg = cv.LoadImage("ponzo.jpg", cv.CV_LOAD_IMAGE_COLOR)     # cimg is a OpenCV image
pimg = Image.fromstring("RGB", cv.GetSize(cimg), cimg.tostring())    # pimg is a PIL image
cimg2 = cv.CreateImageHeader(pimg.size, cv.IPL_DEPTH_8U, 3)      # cimg2 is a OpenCV image
cv.SetData(cimg2, pimg.tostring())

Note: OpenCV stores color image in BGR format. So, the converted PIL image is also in BGR-format. The standard PIL image is stored in RGB format.

# gray image
cimg = cv.LoadImage("ponzo.jpg", cv.CV_LOAD_IMAGE_GRAYSCALE)   # cimg is a OpenCV image
pimg = Image.fromstring("L", cv.GetSize(cimg), cimg.tostring())                 # img is a PIL image
cimg2 = cv.CreateImageHeader(pimg.size, cv.IPL_DEPTH_8U, 1)              # img2 is a OpenCV image
cv.SetData(cimg2, pimg.tostring())




Convert between PIL image and NumPy ndarray

image = Image.open(“ponzo.jpg”)   # image is a PIL image
array = numpy.array(image)          # array is a numpy array
image2 = Image.fromarray(array)   # image2 is a PIL image




Convert between PIL image and PyOpenCV matrix

image = Image.open(“ponzo.jpg”)                  # image is a PIL image
mat = pyopencv.Mat.from_pil_image(image)  # mat is a PyOpenCV matrix
image2 = mat.to_pil_image()                        # image2 is a PIL image




Convert between OpenCV image and NumPy ndarray

cimg = cv.LoadImage("ponzo.jpg", cv.CV_LOAD_IMAGE_COLOR)   # cimg is a OpenCV image
pimg = Image.fromstring("RGB", cv.GetSize(cimg), cimg.tostring())  # pimg is a PIL image
array = numpy.array(pimg)     # array is a numpy array
pimg2 = cv.fromarray(array)    # pimg2 is a OpenCV image




还可以参考这样形式的代码：
PIL.Image转换成OpenCV格式：

    import cv2
    from PIL import Image
    import numpy

    image = Image.open("plane.jpg")
    image.show()
    img = cv2.cvtColor(numpy.asarray(image),cv2.COLOR_RGB2BGR)
    cv2.imshow("OpenCV",img)
    cv2.waitKey()




OpenCV转换成PIL.Image格式：

    import cv2
    from PIL import Image
    import numpy

    img = cv2.imread("plane.jpg")
    cv2.imshow("OpenCV",img)
    image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    image.show()
    cv2.waitKey()



参考文献：
1、http://www.mamicode.com/info-detail-1777172.html 2018.7.12
2、https://blog.csdn.net/u011534057/article/details/77529592/ 2018.7.12
3、https://blog.csdn.net/dcrmg/article/details/78147219 2018.7.12