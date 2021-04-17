# imports for hw1 (you can add any other library as well)
import numpy as np
import matplotlib.pyplot as plt
import cv2


def createGaussianPyramid(im, sigma0, k, levels):
    GaussianPyramid = []
    for i in range(len(levels)):
        sigma_ = sigma0 * k ** levels[i]
        size = int(np.floor(3 * sigma_ * 2) + 1)
        blur = cv2.GaussianBlur(im, (size, size), sigma_)
        GaussianPyramid.append(blur)
    return np.stack(GaussianPyramid)


def displayPyramid(pyramid):
    plt.figure(figsize=(16, 5))
    plt.imshow(np.hstack(pyramid), cmap='gray')
    plt.axis('off')
    plt.show()


def createDoGPyramid(GaussianPyramid, levels):
    # Produces DoG Pyramid
    # inputs
    # GaussianPyramid - A matrix of grayscale images of size
    #                    (len(levels), shape(im))
    # levels          - the levels of the pyramid where the blur at each level is
    #                   outputs
    # DoGPyramid      - size (len(levels) - 1, shape(im)) matrix of the DoG pyramid
    #                   created by differencing the Gaussian Pyramid input
    # DogLevels       - the levels of the pyramid where the blur at each level corresponds
    #                   to the DoG scale
    """
    Your code here
    """
    DoGPyramid = []
    DoGLevels = []
    for i in range(1, len(levels)):
        Dl = GaussianPyramid[i - 1] - GaussianPyramid[i]
        DoGPyramid.append(Dl)
        DoGLevels.append(levels[i])

    return np.stack(DoGPyramid), DoGLevels


def computePrincipalCurvature(DoGPyramid):
    # Edge Suppression
    #  Takes in DoGPyramid generated in createDoGPyramid and returns
    #  PrincipalCurvature,a matrix of the same size where each point contains the
    #  curvature ratio R for the corre-sponding point in the DoG pyramid
    #
    #  INPUTS
    #  DoG Pyramid - size (len(levels) - 1, shape(im)) matrix of the DoG pyramid
    #
    #  OUTPUTS
    #  PrincipalCurvature - size (len(levels) - 1, shape(im)) matrix where each
    #                       point contains the curvature ratio R for the
    #                       corresponding point in the DoG pyramid
    """
    Your code here
    """
    PrincipalCurvature = []
    eps = 1e-6
    for l in range(len(DoGPyramid)):
        # # first derevatives
        # D_x = cv2.Sobel(DoGPyramid[l], cv2.CV_64F, 1, 0)
        # D_y = cv2.Sobel(DoGPyramid[l], cv2.CV_64F, 0, 1)
        # # second derevatives
        # D_xx = cv2.Sobel(D_x, cv2.CV_64F, 1, 0)
        # D_yy = cv2.Sobel(D_y, cv2.CV_64F, 0, 1)
        # D_xy = cv2.Sobel(D_x, cv2.CV_64F, 0, 1)
        #
        # TR_H = D_xx + D_yy  # H.trace()
        # Det_H = D_xx * D_yy - D_xy * D_xy  # numpy.linalg.det(H)
        #
        # R = TR_H ** 2 / Det_H
        Dl = DoGPyramid[l]
        Dxx = cv2.Sobel(Dl, cv2.CV_64F, 2, 0, ksize=3)
        Dxy = cv2.Sobel(Dl, cv2.CV_64F, 1, 1, ksize=3)
        Dyy = cv2.Sobel(Dl, cv2.CV_64F, 0, 2, ksize=3)

        TR_H = Dxx + Dyy  # H.trace()
        Det_H = Dxx * Dyy - Dxy * Dxy  # numpy.linalg.det(H)

        R = TR_H ** 2 / (Det_H + eps)

        PrincipalCurvature.append(R)

    return np.stack(PrincipalCurvature)


def getLocalExtrema(DoGPyramid, DoGLevels, PrincipalCurvature,
                    th_contrast, th_r):
    # Returns local extrema points in both scale and space using the DoGPyramid
    # INPUTS
    #       DoGPyramid         - size (len(levels) - 1, imH, imW ) matrix of the DoG pyramid
    #       DoGlevels          - The levels of the pyramid where the blur at each level is
    #                            outputs
    #       PrincipalCurvature - size (len(levels) - 1, imH, imW) matrix contains the
    #                            curvature ratio R
    #       th_contrast        - remove any point that is a local extremum but does not have a
    #                            DoG response magnitude above this threshold
    #       th_r               - remove any edge-like points that have too large a principal
    #                            curvature ratio
    # OUTPUTS
    #       locsDoG            - N x 3 matrix where the DoG pyramid achieves a local extrema in both
    #                            scale and space, and also satisfies the two thresholds.

    import scipy.ndimage.filters as filters

    x = np.reshape(np.array([]), (-1, 1))
    y = np.reshape(np.array([]), (-1, 1))
    levels = np.reshape(np.array([]), (-1, 1))

    last_k = len(DoGLevels) - 1
    for k in range(len(DoGLevels)):
        l = DoGLevels[k]
        # local maxima in same scale
        neighborhood_size = 8
        max_ = filters.maximum_filter(DoGPyramid[l], neighborhood_size)
        if k == 1:  # first scale
            bool_cond = ((DoGPyramid[l] == max_) & (DoGPyramid[l] > DoGPyramid[l + 1]) & (
                    PrincipalCurvature[l] < th_r) & (np.abs(DoGPyramid[l]) > th_contrast))
        elif k == last_k:  # last scale
            bool_cond = ((DoGPyramid[l] == max_) & (DoGPyramid[l] > DoGPyramid[l - 1]) & (
                    PrincipalCurvature[l] < th_r) & (np.abs(DoGPyramid[l]) > th_contrast))
        else:
            bool_cond = ((DoGPyramid[l] == max_) & (DoGPyramid[l] > DoGPyramid[l - 1]) & (
                    DoGPyramid[l] > DoGPyramid[l + 1]) & (PrincipalCurvature[l] < th_r) & (
                                 np.abs(DoGPyramid[l]) > th_contrast))

        ee = np.argwhere(bool_cond)

        vec = np.reshape(np.ones(ee.shape[0]), (-1, 1)) * l  # level vector
        ee_x = np.reshape(ee[:, 0], (-1, 1))
        ee_y = np.reshape(ee[:, 1], (-1, 1))

        x = np.concatenate((x, ee_x))
        y = np.concatenate((y, ee_y))
        levels = np.concatenate((levels, vec))

    locsDoG = np.reshape(np.stack((x, y, levels)), (3, -1))

    return locsDoG


def DoGdetector(im, sigma0, k, levels, th_contrast=0.03, th_r=12):
    #     Putting it all together
    #     Inputs          Description
    #     --------------------------------------------------------------------------
    #     im              Grayscale image with range [0,1].
    #     sigma0          Scale of the 0th image pyramid.
    #     k               Pyramid Factor.  Suggest sqrt(2).
    #     levels          Levels of pyramid to construct. Suggest -1:4.
    #     th_contrast     DoG contrast threshold.  Suggest 0.03.
    #     th_r            Principal Ratio threshold.  Suggest 12.
    #     Outputs         Description
    #     --------------------------------------------------------------------------
    #     locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
    #                     in both scale and space, and satisfies the two thresholds.
    #     gauss_pyramid   A matrix of grayscale images of size (len(levels),imH,imW)
    """
    Your code here
    """
    GaussianPyramid = createGaussianPyramid(im, sigma0, k, levels)
    DoGPyramid, DoGLevels = createDoGPyramid(GaussianPyramid, levels)
    PrincipalCurvature = computePrincipalCurvature(DoGPyramid)
    locsDoG = getLocalExtrema(DoGPyramid, DoGLevels, PrincipalCurvature, th_contrast, th_r)

    return locsDoG, GaussianPyramid


# example:
if __name__ == '__main__':
    sigma0 = 1
    k = np.sqrt(2)
    levels = [-1, 0, 1, 2, 3, 4]
    th_contrast = 0.03
    th_r = 12

    im = cv2.imread('data/chickenbroth_01.jpg')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im / 255

    locsDoG, GaussianPyramid = DoGdetector(im, sigma0, k, levels, th_contrast=0.03, th_r=12)

    plt.imshow(im, cmap='gray')
    plt.title('Key Points', fontsize=30)
    plt.plot(locsDoG[1, :], locsDoG[0, :], '.')
    plt.show()
