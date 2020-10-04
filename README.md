# Ciona_EM
We have two transforms: one for image and one for contour. Here, we take the example of Series1.7661 xml file.

Step 1: Apply the inverse transform (contour) on the contour points
a =[-12.4219340480873,1.02329692936977,0.0547410605730517,0,0,0]
b =[6.70121729036356,-0.345509738706612,0.972479703389782,0,0,0]


Step 2: Apply the forward transform (image) on the above transformed points
a= [ 7.06307051846487, 0.0288393061798161, -0.365741635336368,0,0,0]
b = [-0.356472580116627,0.330871689624869, -0.284621855197495, 0, 0, 0]

Step 3: Divide the above points by Image mag which is for this example,
0.00284108
