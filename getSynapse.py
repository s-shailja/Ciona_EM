import numpy as np
from tifffile import imread, imsave, imshow
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import xml.etree.ElementTree as ET
import pickle
import ipdb
import os
from math import *
from PIL import Image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def parse_xml(path):
    def parse_coef(str):
        split_str = str.split(' ')
        coef = [float(i) for i in split_str if i]
        return coef

    def parse_points(str):
        split_str = str.split(',  ')
        points = []
        for s in split_str:
            if s:
                double_split_str = s.split(' ')
                points.append([float(i) for i in double_split_str])
        return points

    xml_dict = {'img': {}, 'contours': {}}
    tree = ET.parse(path)
    root = tree.getroot()
    for child in root:
        for gchild in child:
            if gchild.tag == 'Image':
                xml_dict['mag'] = float(gchild.attrib['mag'])
                xml_dict['img']['name'] = gchild.attrib['src']
                xml_dict['img']['xcoef'] = parse_coef(child.attrib['xcoef'])
                xml_dict['img']['ycoef'] = parse_coef(child.attrib['ycoef'])
                xml_dict['img']['dim'] = float(child.attrib['dim'])
            elif gchild.tag == 'Contour':
                contour_name = gchild.attrib['name']
                xml_dict['contours'][contour_name] = {}
                xml_dict['contours'][contour_name]['points'] = parse_points(gchild.attrib['points'])
                xml_dict['contours'][contour_name]['xcoef'] = parse_coef(child.attrib['xcoef'])
                xml_dict['contours'][contour_name]['ycoef'] = parse_coef(child.attrib['ycoef'])
                xml_dict['contours'][contour_name]['dim'] = float(child.attrib['dim'])
    return xml_dict

def Xforward(dim, a, b, x, y):
    if dim == 1:
        return a[0] + x
    elif dim == 2:
        return a[0] + a[1]*x
    elif dim == 3:
        return a[0] + a[1]*x + a[2]*y
    elif dim == 4:
        return a[0] + (a[1] + a[3]*y)*x + a[2]*y
    elif dim == 5:
        return a[0] + (a[1] + a[3]*y + a[4]*x)*x + a[2]*y
    elif dim == 6:
        return a[0] + (a[1] + a[3]*y + a[4]*x)*x + (a[2] + a[5]*y)*y
    return None

def Yforward(dim, a, b, x, y):
    if dim == 1:
        return b[0] + y
    elif dim == 2:
        return b[0] + b[1]*y
    elif dim == 3:
        return b[0] + b[1]*x + b[2]*y
    elif dim == 4:
        return b[0] + (b[1] + b[3]*y)*x + b[2]*y
    elif dim == 5:
        return b[0] + (b[1] + b[3]*y + b[4]*x)*x + b[2]*y
    elif dim == 6:
        return b[0] + (b[1] + b[3]*y + b[4]*x)*x + (b[2] + b[5]*y)*y
    return None

epsilon = 5e-10;
path = "/media/pkao/HD-B1/Dict/"
ser_list = os.listdir(path)
for ser in range(1, 8000):
	try:
		
		print("Processing ..", ser)
		xml_dict = pickle.load(open(path + "Series1." + str(ser) + ".pkl", "rb"))
		contours_list = xml_dict['contours'].keys()
		for contour in contours_list:
			if ("syn" in contour and len(xml_dict['contours'][contour]['points']) == 7):

				
				src = xml_dict['img']['name']
				imag_mag = xml_dict['mag']
				points = xml_dict['contours'][contour]['points']
				try:
					img = imread("/media/pkao/HD-B1/Images/"+src)
					if (len(img.shape) == 3):
						gray = rgb2gray(img)
						img = np.flip(gray.T, 1)
					else:
						img = np.flip(img.T, 1)
					a = xml_dict['contours'][contour]['xcoef']
					b = xml_dict['contours'][contour]['ycoef']
					c_p = []
					for i in points:
					    dim = 6
					    x = i[0]
					    y = i[1]
					    u = x
					    v = y
					    x0 = 0.0           #initial guess of (x,y)
					    y0 = 0.0
					    u0 = Xforward(dim, a, b, x0,y0)          #get forward tform of initial guess
					    v0 = Yforward(dim, a, b, x0,y0)
					    i = 0
					    e = 1.0
					    while (e > epsilon) and (i < 10):
					        i += 1
					        l = a[1] + a[3]*y0 + 2.0*a[4]*x0
					        m = a[2] + a[3]*x0 + 2.0*a[5]*y0
					        n = b[1] + b[3]*y0 + 2.0*b[4]*x0
					        o = b[2] + b[3]*x0 + 2.0*b[5]*y0
					        p = l*o - m*n
					        if abs(p) > epsilon:
					            x0 += (o*(u-u0) - m*(v-v0))/p
					            y0 += (l*(v-v0) - n*(u-u0))/p
					        else:
					            x0 += l*(u-u0) + n*(v-v0)
					            y0 += m*(u-u0) + o*(v-v0)
					        u0 = Xforward(dim, a, b, x0,y0)
					        v0 = Yforward(dim, a, b, x0,y0)
					        e = abs(u-u0) + abs(v-v0)
					    result_x = x0
					    result_y = y0
					    c_p.append([result_x,result_y])
					a = xml_dict['img']['xcoef']
					b = xml_dict['img']['ycoef']
					i_p = []
					for i in c_p:
					    x = i[0]
					    y = i[1]
					    result_x = a[0] + (a[1] + a[3]*y + a[4]*x)*x + (a[2] + a[5]*y)*y
					    result_y = b[0] + (b[1] + b[3]*y + b[4]*x)*x + (b[2] + b[5]*y)*y
					    i_p.append([result_x,result_y])

					centroid_x = int(round(sum(np.array(i_p)[:,0])/7/imag_mag))
					centroid_y = int(round(sum(np.array(i_p)[:,1])/7/imag_mag))
					# print(centroid_x, centroid_y)
					# print("image shape", img.shape)
					c1 = centroid_x-250
					c2 = centroid_x+250
					c3 = centroid_y-250
					c4 = centroid_y+250
					if (centroid_x-250 < 0):
						c1 = 0
						c2 = 500
					if (centroid_x+250 > img.shape[0]):
						c1 = img.shape[0] - 500
						c2 = img.shape[0]
					if(centroid_y-250 < 0):
						c3 = 0
						c4 = 500
					if(centroid_y+250 > img.shape[1]):
						c3 = img.shape[1] - 500
						c4 = img.shape[1]
					try:
						cropped_img = img[c1: c2, c3: c4]
						print(cropped_img.shape)
						imsave(str("/media/pkao/HD-B1/Synapse_data/" + str(ser)+"_" + str(contour.replace("/", "}"))+ ".tif"), cropped_img)
					except:
						print("Unable to save file",str("/media/pkao/HD-B1/Synapse_data/"+ str(ser)+"_" + str(contour.replace("/", "}"))+ ".tif") )
				except:
					print("src image not found", src)

	except:
		print("unable to process ", "Series1." + str(ser))
		continue


