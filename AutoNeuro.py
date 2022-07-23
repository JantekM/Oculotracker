import cv2
import mediapipe as mp
import json
from exif import Image
import numpy as np

eye_landmarks_right = [226, 113, 225, 224, 223, 222, 221, 189, 244, 112, 26, 22, 23, 24, 110, 25]
#[33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7]

eye_landmarks_left =  [446, 342, 445, 444, 443, 442, 441, 413, 464, 341, 256, 252, 253, 254, 339, 255]
# https://github.com/google/mediapipe/blob/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

def face_landmarks_from_photo_batch(image_files: list, min_detection_confidence=0.5, with_ROI=False):
    mp_face_mesh = mp.solutions.face_mesh

    output = []


    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence) as face_mesh:
        for idx, file in enumerate(image_files):
            image = cv2.imread(file)
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.multi_face_landmarks:
                raise Exception(f'Błąd, nie wykryto twarzy na obrazie ćwiczebnym!') # TODO: porządny logging

            landmarks = results.multi_face_landmarks[0].landmark

            if with_ROI:
                #h, w = image.shape[:2]
                coords = get_facemesh_coords(landmarks, image)
                coords = coords[:, [0,1]] #odrzucenie współrzędnej Z
                coords_eyes = (coords[eye_landmarks_right, :], coords[eye_landmarks_left, :])
                ROI_coords_right = np.min(coords_eyes[0][:,0]), \
                                   np.max(coords_eyes[0][:,0]), \
                                   np.min(coords_eyes[0][:,1]), \
                                   np.max(coords_eyes[0][:,1])

                ROI_coords_left =  np.min(coords_eyes[1][:,0]), \
                                   np.max(coords_eyes[1][:,0]), \
                                   np.min(coords_eyes[1][:,1]), \
                                   np.max(coords_eyes[1][:,1])
                ROI_right = image[ROI_coords_right[2]:ROI_coords_right[3], ROI_coords_right[0]:ROI_coords_right[1]]
                ROI_left  = image[ROI_coords_left[2]:ROI_coords_left[3], ROI_coords_left[0]:ROI_coords_left[1]]
                ROIs = (ROI_right, ROI_left, ROI_coords_right, ROI_coords_left)

            else:
                ROIs = None

            with open(file, 'rb') as new_image_file:
                img = Image(new_image_file)
            metadata = json.loads(img.user_comment)


            output.append({"landmarks": landmarks, "ROIs": ROIs, "cursor": (metadata["x"], metadata['y'])})
        return output

def get_facemesh_coords(landmark_list, img):
    h, w = img.shape[:2]  # grab width and height from image
    xyz = [(lm.x, lm.y, lm.z) for lm in landmark_list]

    return np.multiply(xyz, [w, h, w]).astype(int)






face_landmarks_from_photo_batch(['Training Data\\Jantek Mikulski\\2022.07.22\\19.21.49.547198.jpg'], with_ROI=True)