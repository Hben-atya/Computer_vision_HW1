import numpy as np
import matplotlib.pyplot as plt
import cv2

from scipy.spatial.distance import cdist
from my_keypoint_det import *


def makeTestPattern(patchWidth, nbits):
    # Try to implement the first method (uniform dist)
    # assumption: center of patch is top-left - need to fix when used to center!

    # keep uniform random when center is 0,0

    patchIndicesX_x = np.random.randint(low=0, high=patchWidth, size=(nbits))
    patchIndicesX_y = np.random.randint(low=0, high=patchWidth, size=(nbits))
    compareX_ = map(lambda x, y: np.ravel_multi_index((x, y), (patchWidth, patchWidth)), patchIndicesX_x,
                    patchIndicesX_y)
    compareX = np.reshape(np.array(list(compareX_)), (-1, 1))

    patchIndicesY_x = np.random.randint(low=0, high=patchWidth, size=(nbits))
    patchIndicesY_y = np.random.randint(low=0, high=patchWidth, size=(nbits))
    compareY_ = map(lambda x, y: np.ravel_multi_index((x, y), (patchWidth, patchWidth)), patchIndicesY_x,
                    patchIndicesY_y)
    compareY = np.reshape(np.array(list(compareY_)), (-1, 1))

    return compareX, compareY


# additional help function I added
def saveTestPattern():
    from scipy.io import savemat

    patchWidth = 9
    nbits = 256
    compareX, compareY = makeTestPattern(patchWidth, nbits)

    mat = np.append(compareX, compareY, axis=1)
    mdic = {"testPattern": mat}
    savemat("testPattern.mat", mdic)


def computeBrief(GaussianPyramid, locsDoG, patchWidth):
    from scipy.io import loadmat
    mat = loadmat("testPattern.mat")['testPattern']  # array
    compareX, compareY = mat[:, 0], mat[:, 1]

    half_patch = np.int(np.ceil(patchWidth / 2))
    x_valid_range = np.arange(half_patch, GaussianPyramid.shape[2] - half_patch)  # range over cols
    y_valid_range = np.arange(half_patch, GaussianPyramid.shape[1] - half_patch)  # range over rows

    locs_ = []
    desc_ = []

    compareX_fix = np.array(list(map(lambda x: np.unravel_index((x), (patchWidth, patchWidth)), compareX)))
    # fixing locs relative to the center of patch
    compareX_fix_x = compareX_fix[:, 0] - np.int(np.floor(patchWidth / 2))
    compareX_fix_y = compareX_fix[:, 1] - np.int(np.floor(patchWidth / 2))

    compareY_fix = np.array(list(map(lambda x: np.unravel_index((x), (patchWidth, patchWidth)), compareY)))
    compareY_fix_x = compareY_fix[:, 0] - np.int(np.floor(patchWidth / 2))
    compareY_fix_y = compareY_fix[:, 1] - np.int(np.floor(patchWidth / 2))

    for x_keyPoint, y_keyPoint, l_keyPoint in zip(locsDoG[0, :], locsDoG[1, :], locsDoG[2, :]):

        if (x_keyPoint in x_valid_range) & (y_keyPoint in y_valid_range):  # if patch is not exceeds image size

            loc = [x_keyPoint, y_keyPoint, l_keyPoint]
            # test points in the patch around the key point
            inIm_cmpX_x = np.full_like(compareX_fix_x, x_keyPoint) + compareX_fix_x
            inIm_cmpX_y = np.full_like(compareX_fix_y, y_keyPoint) + compareX_fix_y
            inIm_cmpY_x = np.full_like(compareY_fix_x, x_keyPoint) + compareY_fix_x
            inIm_cmpY_y = np.full_like(compareY_fix_y, y_keyPoint) + compareY_fix_y
            descriptor = list(GaussianPyramid[int(l_keyPoint), inIm_cmpX_y, inIm_cmpX_x] <
                              GaussianPyramid[int(l_keyPoint), inIm_cmpY_y, inIm_cmpY_x])

            locs_.append(loc)
            desc_.append(descriptor)
        else:
            continue
    desc = np.stack(desc_)
    locs = np.stack(locs_)

    return locs, desc


def briefLite(im):
    # TODO: check if the following params should be set in other way
    sigma0 = 1
    k = np.sqrt(2)
    patchWidth = 9
    levels = [-1, 0, 1, 2, 3, 4]

    locsDoG, GaussianPyramid = DoGdetector(im, sigma0, k, levels, th_contrast=0.03, th_r=12)

    locs, desc = computeBrief(GaussianPyramid, locsDoG, patchWidth)

    return locs, desc


def briefMatch(desc1, desc2, ratio=0.8):
    #     performs the descriptor matching
    #     inputs  : desc1 , desc2 - m1 x n and m2 x n matrices. m1 and m2 are the number of keypoints in image 1 and 2.
    #                               n is the number of bits in the brief
    #               ratio         - ratio used for testing whether two descriptors should be matched.
    #     outputs : matches       - p x 2 matrix. where the first column are indices
    #                                         into desc1 and the second column are indices into desc2
    D = cdist(np.float32(desc1), np.float32(desc2), metric='hamming')
    # find smallest distance
    ix2 = np.argmin(D, axis=1)
    d1 = D.min(1)
    # find second smallest distance
    d12 = np.partition(D, 2, axis=1)[:, 0:2]
    d2 = d12.max(1)
    r = d1 / (d2 + 1e-10)
    is_discr = r < ratio
    ix2 = ix2[is_discr]
    ix1 = np.arange(D.shape[0])[is_discr]
    matches = np.stack((ix1, ix2), axis=-1)
    return matches


def plotMatches(im1, im2, matches, locs1, locs2):
    fig = plt.figure()
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1] + im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')
    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i, 0], 0:2]
        pt2 = locs2[matches[i, 1], 0:2].copy()
        pt2[0] += im1.shape[1]
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x, y, 'r')
        plt.plot(x, y, 'g.')
    plt.show()


def testMatch(im1, im2):
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im1_gray = im1_gray / 255

    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    im2_gray = im2_gray / 255

    locs1, desc1 = briefLite(im1_gray)
    locs2, desc2 = briefLite(im2_gray)
    # compute feature matches
    matches = briefMatch(desc1, desc2, ratio=0.8)
    plotMatches(im1, im2, matches, locs1, locs2)


def rotateImage(image, theta):
    # rotates an image and calculates the new center pixel
    # INPUTS
    #      image      - HxW image to be rotated
    #      theta      - rotation angle in degrees [0,360]
    # OUTPUTS
    #      image_rot  - H2xW2 rotated image
    #      center_rot - (2,) array of the new center pixel
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -theta, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    image_rot = cv2.warpAffine(image, M, (nW, nH))
    center_rot = (np.array(image_rot.shape[:2][::-1]) - 1) / 2
    return image_rot, center_rot


def checkRotLocs(locs0, locsTheta, matches, theta, c0, cTheta, th_d=9.0):
    # INPUTS
    #     locs0     - m1x3 matrix of keypoints (x,y,l) of the unrotated image
    #     locsTheta - m2x3 matrix of keypoints (x,y,l) of the rotated image
    #     matches   - px2 matrix of matches indexing into locs0 and locsTheta
    #     theta     - rotation angle in degress
    #     c0        - center of the unrotated image
    #     cTheta    - center of the rotated image
    #     th_d      - threshold distance of matched keypoints in pixels
    # OUTPUTS
    #     corrMatch - number of correct matches
    # keep only the matched keypoints (x,y)
    locs0 = locs0[matches[:, 0], :2]
    locsTheta = locsTheta[matches[:, 1], :2]
    # rotate the locations at theta=0 and shift them to the new center
    theta = np.deg2rad(theta)
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    locs0_rot = (rot_mat @ (locs0 - c0).T).T + cTheta
    # count the number of correct matches with a distance threshold Td
    corrMatch = np.sum(np.sqrt(np.sum((locs0_rot - locsTheta) ** 2, 1)) < th_d)
    return corrMatch


# translate openCV to our data structs
def openCV2numpy(kp1, kp2, matches):
    # function transfers OpenCV keypoints and matches to numpy arrays
    # INPUTS
    #     kp1        - keypoints detected for img 1 using orb.detectAndCompute()
    #     kp2        - keypoints detected for img 2 using orb.detectAndCompute()
    #     matches    - matches returned by cv2.BFMatcher()
    # OUTPUTS
    #     locs1      - m1x3 matrix of keypoints (x,y,l)
    #     locs2      - m2x3 matrix of keypoints (x,y,l)
    #     matches_np - px2 matrix indexing into locs1 and locs2

    locs1 = np.array([[kp1[idx].pt[0], kp1[idx].pt[1], float(kp1[idx].octave)]
                      for idx in range(0, len(kp1))]).reshape(-1, 3)
    locs2 = np.array([[kp2[idx].pt[0], kp2[idx].pt[1], float(kp2[idx].octave)]
                      for idx in range(0, len(kp2))]).reshape(-1, 3)
    matches_np = [[mat.queryIdx, mat.trainIdx] for mat in matches]
    matches_np = np.stack(matches_np)
    return locs1, locs2, matches_np


def briefRotTest(im, th_d=9.0):
    theta_gap = 10  # deg
    thetas = np.arange(10, 270, theta_gap)
    c0 = (np.array(im.shape[:2][::-1]) - 1) / 2
    locs0, desc0 = briefLite(im)
    corrMatch_ = []
    for theta in thetas:
        im_rot, cTheta = rotateImage(im, theta)
        diff_center = cTheta - c0
        locsTheta, descTheta = briefLite(im_rot)
        # locsTheta = locsTheta - np.ones_like(locsTheta)*(diff_center[1], diff_center[0], 0)

        matches = briefMatch(desc0, descTheta, ratio=0.8)

        corrMatch = checkRotLocs(locs0, locsTheta, matches, theta, c0, cTheta, th_d)
        corrMatch_.append(corrMatch)

    # bar plot
    plt.figure()
    plt.bar(thetas, corrMatch_)
    plt.show()


def orbTest(im, theta_gap=10, th_d=9.0):
    thetas = np.arange(10, 270, theta_gap)
    c0 = (np.array(im.shape[:2][::-1]) - 1) / 2
    orb0 = cv2.ORB_create()
    kp1, des1 = orb0.detectAndCompute(im, None)
    corrMatch_ = []

    for theta in thetas:
        im_rot, cTheta = rotateImage(im, theta)
        orb = cv2.ORB_create()
        kp2, des2 = orb.detectAndCompute(im_rot, None)
        bf = cv2.BFMatcher()  # (cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        locs0, locsTheta, matches_np = openCV2numpy(kp1, kp2, matches)
        corrMatch = checkRotLocs(locs0, locsTheta, matches_np, theta, c0, cTheta, th_d)
        corrMatch_.append(corrMatch)

    #     cv2.drawMatches()
    # bar plot
    plt.figure()
    plt.bar(thetas, corrMatch_)
    plt.show()


if __name__ == '__main__':
    # saveTestPattern()

    im = cv2.imread('data/model_chickenbroth.jpg')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im / 255

    locs, desc = briefLite(im)

    im_01 = cv2.imread('data/model_chickenbroth.jpg')  # BGR
    im_02 = cv2.imread('data/chickenbroth_01.jpg')

    im = cv2.imread('data/model_chickenbroth.jpg')

    testMatch(im_01, im_02)

    # example:
    im = cv2.imread('data/model_chickenbroth.jpg')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im / 255
    center = (np.array(im.shape[:2][::-1]) - 1) / 2
    theta = 20.0
    im_rot, center_rot = rotateImage(im, theta)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap="gray")
    plt.scatter(center[0], center[1], 200, 'r', '*')
    plt.title("Input", fontsize=30)
    plt.subplot(1, 2, 2)
    plt.imshow(im_rot, cmap="gray")
    plt.scatter(center_rot[0], center_rot[1], 200, 'r', '*')
    plt.title("Rotated", fontsize=30)
    plt.show()

    briefRotTest(im, th_d=9.0)

    # im.shape
    im1 = cv2.imread('data/model_chickenbroth.jpg')
    orbTest(im1, theta_gap=10, th_d=9.0)
