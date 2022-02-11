import os
import cv2
import argparse

from sklearn.svm import LinearSVC
from skimage import feature
from sklearn.metrics import confusion_matrix

# construct the argument parser and parser the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='what folder to use for HOG description', 
					choices=['figures'])
args = vars(parser.parse_args())

args['path']='figures'

images = []
labels = []
# get all the image folder paths
image_paths = os.listdir(f"C:/Users/andru/Documents/IA proyecto final/input/{args['path']}")
for path in image_paths:
	# get all the image names
    all_images = os.listdir(f"C:/Users/andru/Documents/IA proyecto final/input/{args['path']}/{path}")

	# iterate over the image names, get the label
    for image in all_images:
        image_path = f"C:/Users/andru/Documents/IA proyecto final/input/{args['path']}/{path}/{image}"
        image = cv2.imread(image_path)
        image = cv2.resize(image, (128, 256))

        # get the HOG descriptor for the image
        hog_desc = feature.hog(image, orientations=5, pixels_per_cell=(16, 16),
            cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2')
    
        # update the data and labels
        images.append(hog_desc)
        labels.append(path.lower())

# train Linear SVC 
print('Training on train images...')
svm_model = LinearSVC(random_state=10, tol=1e-6, max_iter=1000)
svm_model.fit(images, labels)
# predict on the test images
print('Evaluating on test images...')
# loop over the test dataset
results = []
y_train = []
y_test = []
for (i, imagePath) in enumerate(os.listdir(f"C:/Users/andru/Documents/IA proyecto final/test_images/{args['path']}/")):

    image = cv2.imread(f"C:/Users/andru/Documents/IA proyecto final/test_images/{args['path']}/{imagePath}")
    resized_image = cv2.resize(image, (128, 256))

	# get the HOG descriptor for the test image
    (hog_desc, hog_image) = feature.hog(resized_image, orientations=5, pixels_per_cell=(16, 16),
		cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2', visualize=True)
	# prediction
    pred = svm_model.predict(hog_desc.reshape(1, -1))[0]

	# convert the HOG image to appropriate data type. We do...
	# ... this instead of rescaling the pixels from 0. to 255.
    hog_image = hog_image.astype('float64')
	# show thw HOG image
	#cv2.imshow('HOG Image', hog_image)

	# put the predicted text on the test image
    cv2.putText(image, pred.title(), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
		(0, 255, 0), 2)
	#cv2.imshow('Test Image', image)
    cv2.imwrite(f"C:/Users/andru/Documents/IA proyecto final/outputs/{args['path']}_hog_{i}.jpg", hog_image*255.) # multiply by 255. to bring to OpenCV pixel range
    cv2.imwrite(f"C:/Users/andru/Documents/IA proyecto final/outputs/{args['path']}_pred_{i}.jpg", image)
    cv2.waitKey(0)
    results.append([imagePath[0:len(imagePath)-5],pred.title()])
    y_test.append(pred.title().lower())
    y_train.append(imagePath[0:len(imagePath)-5])               

#print(results)
total_score = 0
secchi3000_score = 0
secchidisk_score = 0
square_score = 0
for i in range(len(results)):
    if results[i][0].lower() == results[i][1].lower():
        total_score += 1
        if i<15:
            secchi3000_score += 1
        if i>=15 and i<30:
            secchidisk_score += 1
        if i>=30:
            square_score += 1
total_score = total_score/45
secchi3000_score = secchi3000_score/15
secchidisk_score = secchidisk_score/15
square_score = square_score/15
print('Total score: '+str(total_score))
#print('Score for secchi3000: '+str(secchi3000_score))
#print('Score for scchidisk: '+str(secchidisk_score))
#print('Score for square: '+str(square_score))
cm = confusion_matrix(y_train, y_test)
print(cm)