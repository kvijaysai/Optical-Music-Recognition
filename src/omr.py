#!/usr/local//bin/python3
# This is just a sample program to show you how to do
# basic image operations using python and the Pillow library.
#
# By Eriya Terada, based on earlier code by Stefan Lee,
#    lightly modified by David Crandall, 2020

#Import the Image and ImageFilter classes from PIL (Pillow)
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import numpy as np
import random
import math
import copy
import sys


STH = 0.8
SPMIN = 1
SPMAX = 40
INC = 20
TEMPS = 11


"""
3.Convolution -2d
"""

def convolution2D(image,kernel):
    conv_image = copy.deepcopy(image)
    #flipping kernel Horizatally + Vertically
    kernel= np.flipud(np.fliplr(kernel))
    kx = (kernel.shape[0]-1)//2
    ky = (kernel.shape[1]-1)//2
    m = conv_image.shape[0]
    n=conv_image.shape[1]
    
    # Mirror padding to the input image
    #creating a matrix of size ( m+ 2*kx ,n + 2*ky) with paading extension
    image_padded = np.zeros((image.shape[0] + 2*kx, image.shape[1] + 2*ky))
    image_padded[kx:-kx, ky:-ky] = conv_image
    
    #Left mirror padding
    image_padded[kx:-kx,0:ky] =image_padded[kx:-kx,2*ky-1:ky-1:-1]
    
    #Right mirror padding 
    image_padded[kx:-kx,n+ky:] =image_padded[kx:-kx, n+ky-1:n-1:-1]
    
    #Top mirror padding
    image_padded[0:kx,:] =image_padded[2*kx-1:kx-1:-1,:]
    
    #Bottom mirror padding 
    image_padded[m+kx:,:] =image_padded[m+kx-1:m-1:-1,:]
    
    for i in range(kx,m+kx):
        for j in range(ky,n+ky):
            conv_image[i-kx,j-ky] = (kernel*image_padded[i-kx:i+kx+1,j-ky:j+ky+1]).sum()
    return conv_image  

"""
4. Convolution -Seperable
"""
def seperableConvolution(I, hx, hy):
    hx_f = np.flip(hx,0)
    hy_f = np.flip(hy,1)
    
    kx = hx.shape[0]//2
    ky = hy.shape[1]//2
    m = I.shape[0]
    n = I.shape[1]
    
    # Mirror padding to the input image
    #creating a matrix of size ( m+ 2*kx ,n + 2*ky) with paading extension
    image_padded = np.zeros((I.shape[0] + 2*kx, I.shape[1] + 2*ky))
    image_padded[kx:-kx, ky:-ky] = I
    #Left mirror padding
    image_padded[kx:-kx,0:ky] =image_padded[kx:-kx,2*ky-1:ky-1:-1]
    #Right mirror padding 
    image_padded[kx:-kx,n+ky:] =image_padded[kx:-kx, n+ky-1:n-1:-1]
    #Top mirror padding
    image_padded[0:kx,:] =image_padded[2*kx-1:kx-1:-1,:]
    #Bottom mirror padding 
    image_padded[m+kx:,:] =image_padded[m+kx-1:m-1:-1,:]
    conv_imagex = copy.deepcopy(image_padded)
    conv_imagey = copy.deepcopy(image_padded)
    
    #convolution with hx
    for i in range(m):
        for j in range(ky,n+ky):
            conv_imagex[i,j-ky] = (hx*image_padded[i,j-ky:j+ky+1]).sum()
    
    #convolution with hy      
    for i in range(kx,m+kx):
        for j in range(n):
            conv_imagey[i-kx,j] = (hy*conv_imagex[i-kx:i+kx+1,j]).sum()
    
    return conv_imagey



"""
5.+ Hamming distance

"""

def detectBoundaries(image,template,symbol):
    
    y1 = 0 
    y2 = template.shape[0] 
    x1 =0
    x2 = template.shape[1] 
    alpha = 0.55
    
    if symbol == "eighth_rest":
        alpha = 0.53
    elif symbol == "quarter_rest":
        alpha = 0.39
        
    region = {}
    final_regions ={}
    while True:
        region[((x1,y1),(x2,y2))] = np.sum(image[y1:y2,x1:x2])
        x1 +=template.shape[1] 
        x2 +=template.shape[1] 
        if x2 >= image.shape[1]:
            x1 = 0
            x2  = template.shape[1] 
            y1 += template.shape[0] 
            y2 += template.shape[0] 
            if y2 >= image.shape[0]:
                break
    sorted_regions = sorted(region.items(), key=lambda item: item[1],reverse=True)
    sum_value = 0
    for key,value in sorted_regions:
        sum_value += value
        if value >= sorted_regions[0][1]*alpha:
             final_regions[key] = value
            
    return final_regions,sum_value
    

def scoreHamming(image,template):
    
    #convolve = convolution2D(image,template)
    convolve = np.convolve(image.flatten(),template.flatten(),mode="same")
    convolve = np.reshape(convolve,(image.shape[0],image.shape[1]))
    template_sum = np.sum(template)
    length = len(template.flatten())
    final_array = np.zeros((image.shape[0],image.shape[1]))
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            array = np.zeros((0,0))
            if i + template.shape[0] < image.shape[0] and j + template.shape[1] < image.shape[1]:
                array = image[i:i+template.shape[0]+1,j:j+template.shape[1]+1]
            elif i + template.shape[0] >= image.shape[0] and j + template.shape[1] < image.shape[1]:
            #    print("enter1")
                array = image[i:image.shape[0],j:j+template.shape[1]+1]
            elif i + template.shape[0] < image.shape[0] and j + template.shape[1] >= image.shape[1]:
             #   print("enter2")
                array = image[i:i+template.shape[0]+1,j:image.shape[1]]   
            final_array[i][j] = 2*convolve[i][j] + (1*length - np.sum(array)  - template_sum )  
    
    return final_array


"""
6.+  Scoring function

"""
def computeGamma(value):
    
    if value == 0:
        return math.inf
    else:
        return 0

def computeD(image,template):
    final_array = np.zeros((image.shape[0],image.shape[1]))
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            min_value = math.inf
            for k in range(0,image.shape[0]):
                for l in range(0,image.shape[1]):
                    gamma = computeGamma(image[k][l])
                    if gamma != math.inf:
                        d = gamma + math.sqrt(math.pow((i-k),2) + math.pow((j-l),2))
                        min_value = min(min_value,d)
            final_array[i][j] = min_value
    return final_array
            
    
def generateSobelVectors(flag):
    hx = np.transpose(np.array([1,2,1]))
    hx = hx[:,None]
    hy = np.array([-1,0,1])
    hy = hy[None,:]
    
    if flag:
        hx,hy = np.transpose(hy),np.transpose(hx)
    
    return hx,hy
    

def scoringFunction(image,template):
    
    #generate sobel vectors
    hx,hy = generateSobelVectors(False)
    image_edge_map1 = seperableConvolution(image,hx,hy)
    template_edge_map1 = seperableConvolution(template,hx,hy)
    
    hx,hy = generateSobelVectors(True)
    image_edge_map2 = seperableConvolution(image,hx,hy)
    template_edge_map2 = seperableConvolution(template,hx,hy)
    
    image_edge_map = (np.sqrt(np.square(image_edge_map1/255) + np.square(image_edge_map2/255))*255).astype('uint16')
    template_edge_map = (np.sqrt(np.square(template_edge_map1/255) + np.square(template_edge_map2/255))*255).astype('uint16')
    
    D = computeD(image_edge_map,template_edge_map)
    #Convolution
    #convolvedImage = convolution2D(D,template_edge_map)
    convolvedImage = np.convolve(D.flatten(),template_edge_map.flatten(),mode="same")
    convolvedImage = np.reshape(convolvedImage,(D.shape[0],D.shape[1]))
    
    return convolvedImage,image_edge_map,template_edge_map


"""
7. Hough Transform
"""
def Convolution_2d_hough(accumulator):
    conv_image = copy.deepcopy(accumulator)
    kernel = np.ones((3,3))
#    kernel = np.array([[0,1,0],[0,1,0],[0,1,0]])
#    kernel[1,1] = 2
    #flipping kernel Horizatally + Vertically
    kx = (kernel.shape[0]-1)//2
    ky = (kernel.shape[1]-1)//2
    m = conv_image.shape[0]
    n=conv_image.shape[1]
    
    for i in range(kx,m+kx-10):
        for j in range(ky,n+ky-10):
            conv_image[i,j] = (kernel*accumulator[i-kx:i+kx+1,j-ky:j+ky+1]).sum()
    return conv_image  

def houghTransform(binary_edge_map, testing=False):
   
    emap_shape = binary_edge_map.shape

    # to get the row nubers of the image that have long horizontal lines
    row_sums = np.sum(binary_edge_map, axis=1)

    selected = []
    iter = 0
    for i in range(emap_shape[0]):
        if row_sums[i]/emap_shape[1] >= STH:
            if iter == 0:
                selected.append(i)
                iter = 1
            else:
                if i - selected[-1] > 3:
                    selected.append(i)
    #selected.pop()
    #selected.pop()
#    setting up the accumulator matrix

#    selected = [31,43,55,67,79,152,164,176,188,200]
    frowc_min = 0
    frowc_max = emap_shape[0]
    space_min = SPMIN
    space_max = SPMAX

    accumulator = np.zeros((frowc_max,SPMAX-SPMIN+1))

    # voting
    for y in selected:
        for pos in range(5):
            if pos == 0:
                accumulator[y] += INC
            else:
                b = y/pos
                for space in range(SPMIN, SPMAX-SPMIN+1+SPMIN):
                    row = int(np.rint((1-space/b)*y))
                    accumulator[row, space-SPMIN] += INC

    # getting max hits
    accumulator=Convolution_2d_hough(accumulator)
    hits = np.argwhere((accumulator == np.max(accumulator)))
    
    
    # calculating scaling factor
    scale = np.average(hits[:,1])/TEMPS

    # testing

    
    imp_hits = []
    hits_map={}
    min_val = 0
    flag = 1
    for ind,val in enumerate(hits):
        if ind == 0:
            imp_hits.append(val)
            hits_map[val[0]]=flag
            flag=2
            min_val = val
        if val[0] > min_val[0]+4*min_val[1]:
            imp_hits.append(val)
            if flag ==2:
                hits_map[val[0]]=2
                flag=1
            else:
                hits_map[val[0]]=1
                flag=2
            min_val = val
    imp_hits = np.array(imp_hits)
    
    if testing == True:
        print(np.histogram(accumulator, bins=80))
        # hough_img = Image.fromarray(accumulator).convert("RGB")
        with Image.open(binary_edge_map_img) as im:
            im = im.convert("RGB")
            # print(np.array(im).shape)
        draw = ImageDraw.Draw(im)
        for i in imp_hits[:,0]:
            draw.line(((0, i), (emap_shape[1], i)), fill=(255, 255, 0), width = 1)
        im.show()
        # hough_img.show()
        print(selected)
    return scale,hits_map
    
"""
8. Image Pipeline for OMR
"""


def drawBoundaries(image,region):
    draw1 = ImageDraw.Draw(image)
    for key, value in region.items():
        draw1.rectangle(((key[0][0],key[0][1]), (key[1][0],key[1][1])),outline="red")
    return image
    

def scaleTemplate(template,scaling_factor):
 
    template_array = template.convert('L')
    template_array = np.asarray(template_array)
    maxsize = (int(template_array.shape[0]*scaling_factor),int(template_array.shape[1]*scaling_factor))
    template.thumbnail(maxsize,Image.ANTIALIAS)
    
    return template
  
def classfiySymbols(regions,mapping,staffMap,spacingFactor):
    mapping = { 1: {-3:"A" ,-2:"B" ,-1:"C",0:"D"} ,2:{-3:"A" ,-2:"B" ,-1:"C",0:"D"}}
    finalMapping = {}
    
    for key,value in regions.items():
        midy = (key[0][1] + key[1][1])/2
        min_value = math.inf
        cord = 0
        for cordinate,partition in staffMap.keys():
            score = int((midy - cordinate)/(spacingFactor/2))
            if score < min_value :
                min_value = score
                cord = cordinate
        if staffMap[cord] in mapping.keys() :
            finalMapping[key] = mapping[staffMap[cord]][min_value]
        else:
             finalMapping[key] = "NA"
        
    return finalMapping
            

def imagePipeline(image,template,flag,symbol):
    final_dict={}
    
    #1
    image2 = image.convert('L')
    image2 = np.asarray(image2)
    template2 = template.convert('L')
    template2 = np.asarray(template2)
    
    #2
    #template1 = template.convert('1')
    hx,hy = generateSobelVectors(False)
    image_edge_map1 = seperableConvolution(image2,hx,hy)   
    hx,hy = generateSobelVectors(True)
    image_edge_map2 = seperableConvolution(image2,hx,hy)
    image_edge_map = (np.sqrt(np.square(image_edge_map1/255) + np.square(image_edge_map2/255))*255).astype('uint16')
    binary_edge_map_img = Image.fromarray(image_edge_map1)
    binary_edge_map_img.convert("L").save("edge1.png")
    binary_edge_map = np.array(binary_edge_map_img.convert("1"))
    scaling_factor,staffMap = houghTransform(binary_edge_map)

    #3
    if 0.8 <= scaling_factor <= 1.25:
        template = scaleTemplate(template,scaling_factor)
    #4
    #Hamming distance
    image1= image.convert('1')
    image1 = np.asarray(image1)
    template1 = template.convert('1')
    template1 = np.asarray(template1)
    final_array1 = scoreHamming(image1,template1)
    image11 = Image.fromarray(final_array1)
    image11.convert("L").save("hamming1.png")
    regions1,sum_value1 = detectBoundaries(final_array1,template1,symbol)
    """
    
       This method is used to classify symbols based on the staff spacing which is the spacing factor
       ,regions, staffMap which consists of first row cordinates of every staff in the image.
    
            regions1 = classfiySymbols(regions1,staffMap,scaling_factor)

        
    """

    
    #Scoring function
    if flag:
        template2 = template.convert('L')
        template2 = np.asarray(template2)
        final_array2 = scoringFunction(image2,template2) 
        regions2,sum_value2 = detectBoundaries(final_array2,template2,symbol)
        final_dict["final_array2"] = final_array2
        final_dict["sum_value2"] = sum_value2
        final_dict["image_copy2"] = image_copy2



    final_dict["final_array1"] = final_array1
    final_dict["regions1"] = regions1
    final_dict["sum_value1"] = sum_value1
    final_dict["image_copy1"] = image_copy1
    final_dict["template"] = template
    

    return final_dict


if __name__ == '__main__':
    #Load an image (this one happens to be grayscale)
    im = Image.open("example.jpg")
    #Check its width, height, and number of color channels
    print("Image is %s pixels wide." % im.width)
    print("Image is %s pixels high." % im.height)
    print("Image mode is %s." % im.mode)

    #pixels are accessed via a (X,Y) tuple
    print("Pixel value is %s" % im.getpixel((10,10)))

    #pixels can be modified by specifying the coordinate and RGB value
    im.putpixel((10,10), 20)
    print("New pixel value is %s" % im.getpixel((10,10)))

    #Create a new blank color image the same size as the input
    color_im = Image.new("RGB", (im.width, im.height), color=0)

    # Loops over the new color image and fills in any area that was white in the 
    # first grayscale image  with random colors!
    for x in range(im.width):
        for y in range(im.height):

            if im.getpixel((x,y)) > 200:
                R = random.randint(0,255)
                G = random.randint(0,255)
                B = random.randint(0,255)
                color_im.putpixel((x,y), (R,G,B))
            else:
                color_im.putpixel((x,y), (0,0,0))

    #Save the image
    color_im.save("output.png")

    # Using Pillow's code to create a convolution kernel and apply it to our color image
    # Here, we are applying the box blur, where a kernel of size 3x3 is filled with 1
    # and the result is divided by 9
    # Note: The assignment requires you to implement your own convolution, but
    #   there's nothing stopping you from using Pillow's built-in convolution to check
    #   that your results are correct!
    result = color_im.filter(ImageFilter.Kernel((3,3),[1,1,1,1,1,1,1,1,1],9))
    # Draw a box and add some text. Just for fun!
    draw = ImageDraw.Draw(result)
    #font = ImageFont.truetype("/usr/share/fonts/msttcorefonts/arial.ttf", 16)
    #draw.text((0, 0),"Hello!",(0,255,0), font=font)
    draw.rectangle(((100,100), (200,200)), (0,255,0))
    
    result.save("convolved.png")
    
    
    image_file = str(sys.argv[1])
    image = Image.open(image_file)
    
    
    
    # Set the flag to true , if we would like to compare both hamming and eculedian distance scores.
    flag = False
    
    
    
    
    f2 = None
    image_copy1 = copy.deepcopy(image)
    image_copy2 = copy.deepcopy(image)
    template_file = "template"
    if flag :
        f2 = open("detected1.txt", "a")
    f1 = open("detected.txt","a")
    symbol = ""
    for i in range(1,4):
        file_name = template_file+str(i)+".png"
        template = Image.open(file_name)
        
        if i == 1:
            symbol = "filled_note"
        elif i == 2:
            symbol = "eighth_rest"
        elif i ==3:
            symbol = "quarter_rest"
            
        final_dict = imagePipeline(image,template,flag,symbol)
        final_template = final_dict["template"]
        final_template = final_template.convert('L')
        final_template = np.asarray(final_template)
        regions1 = final_dict["regions1"]
        final_array1  = final_dict["final_array1"]
        sum_value1 = final_dict["sum_value1"]
        
        image_copy1 = drawBoundaries(image_copy1,regions1)

        # If flag is set to True , even the eculedian distance scoring function will be 
        # used to score the template regions on the image .
        if flag :
            regions2 = final_dict["regions2"]
            final_array2 = final_dict["final_array2"]        
            sum_value2= final_dict["sum_value2"]
            image_copy2 = drawBoundaries(image_copy2,regions2)
            
            for key,value in regions2.items():
                f2.write(str(key[0][0]) +" " + str(key[0][1]) +" "+str(final_template.shape[0])+
                " "+str(final_template.shape[1])+" "+symbol +"  A  "+ str(abs(value/sum_value2)*100))
                f2.write("\n")
            
        
        sorted_regions1 = sorted(regions1.items(), key=lambda item: item[1],reverse=True)
        #Writing the results to a file .
        for key,value in regions1.items():
            f1.write(str(key[0][0]) +" " + str(key[0][1]) +" "+str(final_template.shape[0])+
                " "+str(final_template.shape[1])+" "+symbol +"  A  "+ str(abs(value/sorted_regions1[0][1])*100))
            f1.write("\n")
            
    f1.close()
    if flag :
        f2.close()

    image_copy1.save('detected.png')

    if flag :    
        image_copy2.save('detected1.png')


   
        
    
    #template = template.convert('1')
    #image= image.convert('1')
    #template = np.asarray(template)
    #image = np.asarray(image)
    #final_array = scoreHamming(image,template)
    #regions_list = detect_regions(final_array)
    #final_image =Image.fromarray(final_array)
    #draw = ImageDraw.Draw(final_image)
    #draw.rectangle(((21,143), (24,624)), 150)
    #draw.rectangle(((220,1257), (229,1273)),150)
    
    

    
        
        
    
    
    
    

    
