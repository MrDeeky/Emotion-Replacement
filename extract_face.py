import cv2 as cv
import numpy as np
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from model import *
from emotion_recognition import *

def getFaces(face_cascade, img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_regions = face_cascade.detectMultiScale(img)
    return face_regions

def getEyes(eye_cascade, face):
    face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
    eye_regions = eye_cascade.detectMultiScale(face)
    return eye_regions

def findLeftEye(eye_locations):
    assert(len(eye_locations) == 2)
    if eye_locations[0][0] <= eye_locations[1][0]:
        return eye_locations[0]
    return eye_locations[1]

def findRightEye(eye_locations):
    assert(len(eye_locations) == 2)
    if eye_locations[0][0] >= eye_locations[1][0]:
        return eye_locations[0]
    return eye_locations[1]

def getNose(nose_cascade, face):
    nose_region = nose_cascade.detectMultiScale(face)
    return nose_region

def getDistance(p1, p2):
    return np.sqrt(np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1]))

def getCosine(a, b, c):
    return (np.square(b) + np.square(c) - np.square(a)) / (2*b*c)

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    if angle == 0:
        return mat
    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origin) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv.warpAffine(mat, rotation_mat, (bound_w, bound_h), borderMode=cv.BORDER_CONSTANT)
    return rotated_mat

def resize_img(img, new_shape=(128,128)):
    return cv.resize(img, new_shape)

def preProcessFaces(img, face_locations):
    # Stores tuples of np arrays and radian angles, so we can store rotated faces and info to reverse the transformation
    face_imgs = []
    
    for (x,y,w,h) in face_locations:
        #img = cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face = np.copy(img[y:y+h, x:x+w])

        eye_locations = getEyes(eye_cascade, face)

        if (len(eye_locations) != 2): # Can't classify finer grained features, just pass on the face without further processing
            print('no eye keypoints detected, skipping rotation correction')
            face_imgs.append((resize_img(face), 0, face, (x,y,w,h)))
            continue
        
        # for (i,j,ew,eh) in eye_locations:
        #     cv.rectangle(face,(i,j),(i+ew,j+eh),(255,0,0),2)
        nose_locations = getNose(nose_cascade, face)

        # After processing, nose, left_eye, right_eye should be coordinates for rectangles
        nose = None
        left_eye = findLeftEye(eye_locations)
        right_eye = findRightEye(eye_locations)

        #Some heuristic calculations based on layout of human face
        #basically accepting a feature as a nose if it falls roughly in between the eyes
        for (i, j, nw, nh) in nose_locations:
           if (j < right_eye[0] + right_eye[2] and j > left_eye[0]):
                # cv.rectangle(face,(i,j),(i+ew,j+eh),(255,0,0),2)
                nose = np.array([i, j, nw, nh])
                break

        # Get centre points of nose, left eye, right eye (in coordinate space of face), use it to calculate angle of rotation required to
        # straighten out face
        nose_centre = ((2*nose[0] + nose[2]) // 2, (2*nose[1] + nose[3]) // 2)
        left_eye_centre = ((2*left_eye[0] + left_eye[2]) // 2, (2*left_eye[1] + left_eye[3]) // 2)
        right_eye_centre = ((2*right_eye[0] + right_eye[2]) // 2, (2*right_eye[1] + right_eye[3]) // 2)
        midpoint_eye = ((left_eye_centre[0] + right_eye_centre[0]) // 2, (left_eye_centre[1] + right_eye_centre[1]) // 2)

        # Get top centre of the face bounding box (in coordinate space of face)
        face_top_centre = (w // 2, 0)

        # Calculate triangle dimensions between nose centre, midpoint_eye, face_top_centre
        a = getDistance(face_top_centre, midpoint_eye)
        b = getDistance(face_top_centre, nose_centre)
        c = getDistance(midpoint_eye, nose_centre)

        # apply cosine rule to get angle of rotation
        rotationAngleRad = np.arccos(getCosine(a, b, c))


        # get direction of rotation, based on midpoint of the eyes, if its to the left of top centre of face bounding box
        # its a clockwise rotation (needs to be negative rotation value by openCV conventions) 
        # otherwise its a counter-clockwise rotation (positive rotation value)

        if (midpoint_eye[0] < face_top_centre[0]):
            rotationAngleRad = -rotationAngleRad
        
        rotationAngleDeg = np.degrees(rotationAngleRad)
        print('Angle of rotation', rotationAngleDeg)

        # cv.circle(face, nose_centre, 1, (0,255,0), 2)
        # cv.circle(face, left_eye_centre, 1, (0,0,255), 2)
        # cv.circle(face, right_eye_centre, 1, (0,0,255), 2)
        # cv.circle(face, midpoint_eye, 1, (0,255,0), 2)
        # cv.circle(face, face_top_centre, 1, (0,255,0), 2)
        # cv.imshow("face", face)
        # Only need to rotate faces that have more obvious skewed orienation, minute rotation is OK to pass into the models
        # store rotated face, angle of rotation (if any), original face, and original face coords
        # this will allow us to undo the rotation and merge a changed face back into the original photo
        if (np.abs(rotationAngleDeg) >= 15):
            print("Face is relatively frontal and non-rotated, skipping rotation")
            rotated_face = rotate_image(face, rotationAngleDeg)
            face_imgs.append((resize_img(rotated_face), rotationAngleDeg, face, (x,y,w,h)))
            # cv.imshow("rotated_face %d" % rotationAngleDeg, rotated_face)
        else:
            face_imgs.append((resize_img(face), 0, face, (x,y,w,h)))
    return face_imgs

if __name__ == "__main__":
    face_cascade = cv.CascadeClassifier()
    face_cascade.load("./Weights/haarcascade_frontalface_default.xml")

    eye_cascade = cv.CascadeClassifier()
    eye_cascade.load("./Weights/haarcascade_eye_tree_eyeglasses.xml")

    nose_cascade = cv.CascadeClassifier()
    nose_cascade.load("./Weights/haarcascade_mcs_nose.xml")

    # Load Generator network
    gan = StarGAN(Config())
    G = gan.build_generator()
    G.load_weights('./Weights/G_weights_v4.hdf5')
    G.trainable = False

    classifier = build_cnn_project()
    classifier.load_weights('./Weights/emotion_recognition.h5')

    if (len(sys.argv) == 1):
        print('Pass in a filepath to the image you want to process')
        print ('e.g python extract_face.py img.jpg')
        exit(0)
    
    img = cv.imread(sys.argv[1])
    #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # face locations given by x y coordinate + width and height for bounding box drawing (x y is top left of box)
    face_locations = getFaces(face_cascade, img)

    face_imgs = preProcessFaces(img, face_locations)


    cv.imshow("full picture", img)
    if (len(face_imgs) == 0):
        print('no faces detected :(')
        exit(0)
    #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # TODO face_imgs should now store all faces, resized to 128x128, need to pass them into the models
    # then need to resize results from 128x128 to original face dimensions, and proceed with image merge.
    # Models will output an image that is 128x128 (should also be same orientation)
    for i in range(len(face_imgs)):
        face = face_imgs[i]

        classifier_input = cv.cvtColor(face[0], cv.COLOR_BGR2GRAY)
        classifier_input = np.expand_dims(classifier_input, axis=-1)
        classifier_input = np.expand_dims(classifier_input, axis=0)
        emotion_vector = classifier.predict(classifier_input)[0]
        

        emotion_vector = np.squeeze(emotion_vector)

        
        gan_emotion_vector = swap_emotion(emotion_vector)
        face_tensor = np.expand_dims(face[0]/127.5 - 1, axis=0)
        pred = G.predict([face_tensor, gan_emotion_vector])
        
        gan_face = np.squeeze(pred)*127.5 + 127.5
        
        cv.imshow("ganface %d"%i, gan_face.astype(np.uint8))

        original_face_x = face[3][0]
        original_face_y = face[3][1]
        original_face_width = face[3][2]
        original_face_height = face[3][3]
        resized_face = None
        # If we didn't rotate, just resize the transformed face to original face size
        if (face[1] == 0):
            resized_face = resize_img(gan_face, (original_face_width, original_face_height))
        else:
            # If we rotated, undo the rotation, then resize, face needs to be retrieved again since rotation
            # causes an image resizing
            unrotated_face = rotate_image(gan_face, -face[1]).astype(np.uint8)
            faces2 = getFaces(face_cascade, unrotated_face)
            for (x,y,w,h) in faces2:
                resized_face = resize_img(unrotated_face[y:y+h, x:x+w], (original_face_width, original_face_height))
        #cv.imshow("retransformed face %d" % i, resized_face)

        # clone face back into photo
        clone_mask = np.ones(resized_face.shape, dtype=np.uint8) * 255
        clone_origin = ((original_face_width + 2*original_face_x) // 2, (original_face_height + 2*original_face_y) // 2)
        img = cv.seamlessClone(resized_face, img, clone_mask, clone_origin, cv.NORMAL_CLONE)
    
    #img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imshow("cloned_img", img)
    cv.waitKey(0)
    cv.destroyAllWindows()