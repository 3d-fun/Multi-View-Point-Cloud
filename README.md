# Stereoscopic-Point-Cloud

## Description
Takes a set of stereo images and camera matricies and produces a 3D point cloud of the model.
A reference image is compared with a subset of nearby images. We perform ray marching and record depth values with similar pixel values between images.
We use normalized cross correlation as our Photo-consistency measure. Points are recorded with a position, normal and color.

Based on the work presented in [1].

## API

```
gs::computePointCloud(std::vector<gs::StereoImage*>& images, float* globalRange, int rayMarchIterations, float nccThresh, std::vector<PointCloud*>& pointCloud)
```
#### Parameters
+ std::vector<gs::StereoImage*>& images: *vector of stereo images*
+ **float[] globalRange**: *Global range of the object. {minX, minY, minZ, maxX, maxY, maxZ}*
+ **int rayMarchIterations**: *number of ray marching iterations*
+ **float nccThresh**: *normalized cross correlation threshold*
+ std::vector<PointCloud*>& pointCloud: *output point cloud.*

## Classes

```
class StereoImage
{
	cv::Mat image;
	cv::Mat cameraMatrix;
	cv::Mat cameraMatrixInv;
};

class PointCloud
{
	float position[3];
	float normal[3];
	float color[3];
}; 
```

## Example

Using the Middlebury Computer Vision Datasets : http://vision.middlebury.edu/mview/data/

```
#include <opencv2\opencv.hpp>
#include "PointCloud.h"

const char* DINOSAUR_IMAGE_0 = "../images/dino/dinoSR0001.png";
const char* DINOSAUR_IMAGE_1 = "../images/dino/dinoSR0002.png";
const char* DINOSAUR_IMAGE_2 = "../images/dino/dinoSR0003.png";
const char* DINOSAUR_IMAGE_3 = "../images/dino/dinoSR0004.png";
const char* DINOSAUR_IMAGE_4 = "../images/dino/dinoSR0005.png";
const char* DINOSAUR_IMAGE_5 = "../images/dino/dinoSR0006.png";
const char* DINOSAUR_IMAGE_6 = "../images/dino/dinoSR0007.png";
const char* DINOSAUR_IMAGE_7 = "../images/dino/dinoSR0008.png";
const char* DINOSAUR_IMAGE_8 = "../images/dino/dinoSR0009.png";
const char* DINOSAUR_IMAGE_9 = "../images/dino/dinoSR0010.png";
const char* DINOSAUR_IMAGE_10 = "../images/dino/dinoSR0011.png";
const char* DINOSAUR_IMAGE_11 = "../images/dino/dinoSR0012.png";
const char* DINOSAUR_IMAGE_12 = "../images/dino/dinoSR0013.png";
const char* DINOSAUR_IMAGE_13 = "../images/dino/dinoSR0014.png";
const char* DINOSAUR_IMAGE_14 = "../images/dino/dinoSR0015.png";
const char* DINOSAUR_IMAGE_15 = "../images/dino/dinoSR0016.png";

const double CAMERA_MATRIX_0[] = { -604.322415634554, 3133.97313889453,	933.850029289689, 178.710771150066,	-3086.02553433880,	-205.839787774483,	 -1238.24718105246,		37.1273242565340,	-0.403315364597785,		-0.239841305752123,		0.883069362014872,		0.673097816109000,	0,	0,	0,	1 };
const double CAMERA_MATRIX_1[] = { -992.931129840735, 3110.00105339032, 633.285380588841, 161.355839060032, -2333.79749833643, -243.095767781870, -2365.05074736584, 27.3536918061944, -0.716012081871973, -0.258781378563770, 0.648350496196070, 0.659314086504000, 0, 0, 0, 1 };
const double CAMERA_MATRIX_2[] = { -1229.33222645543, 3083.29199398620, 202.784015180048, 144.512313700642, -1194.86337984720, -251.939272683925, -3099.67571238765, 37.2127005841443, -0.909078106128552, -0.280020268237339, 0.308489954619108, 0.645855774902000, 0, 0, 0, 1 };
const double CAMERA_MATRIX_3[] = { -1274.38302682699, 3058.26837726277, -286.372770671148, 130.969102187723, 142.195072381436, -230.906017942663, -3320.48478531382, 65.0719215619694, -0.950546067273876, -0.300041294667216, -0.0802390382859129, 0.634951273531000, 0, 0, 0, 1 };
const double CAMERA_MATRIX_4[] = { -1120.62414714893, 3039.07354922935, -753.191650488463, 122.968654455760, 1455.99114550109, -183.478637650349, -2990.91699951521, 106.318495446757, -0.833549816339950, -0.315529427852728, -0.453471732376838, 0.628406121531000, 0, 0, 0, 1 };
const double CAMERA_MATRIX_5[] = { -793.514569847014, 3028.88573797002, -1120.37800764765, 121.835663156526, 2528.99001351251, -117.510030084409, -2165.54130401327, 154.122925289393, -0.577461278399623, -0.323920184289549, -0.749409260365608, 0.627304047594000, 0, 0, 0, 1 };
const double CAMERA_MATRIX_6[] = { -347.216250235622, 3029.39181574115, -1327.13405931449, 127.757726514203, 3183.52725686700, -43.9231071878404, -981.021261886720, 200.569876431686, -0.224682929082709, -0.323824243575083, -0.919051039469502, 0.631827530638000, 0, 0, 0, 1 };
const double CAMERA_MATRIX_7[] = { 144.373896833833, 3040.50798750181, -1339.22567626718, 139.754284682889, 3311.22630139023, 25.0977997602228, 366.513379459067, 237.968783520935, 0.166373110176873, -0.315257491323268, -0.934308225176181, 0.641227584108000, 0, 0, 0, 1 };
const double CAMERA_MATRIX_8[] = { 599.859653532394, 3060.39366644722, -1154.65076002339, 155.838978766586, 2890.94307146041, 78.1243885242947, 1653.94133931309, 260.127232803659, 0.530956797612306, -0.299638388088152, -0.792654572363278, 0.653947771602000, 0, 0, 0, 1 };
const double CAMERA_MATRIX_9[] = { 943.822873074084, 3085.75623358418, -803.970745099048, 173.348546957734, 1992.26697104984, 106.376654469423, 2668.09364493643, 263.376285885590, 0.808701314612015, -0.279553104355040, -0.517544724394500, 0.667881917561000, 0, 0, 0, 1 };
const double CAMERA_MATRIX_10[] = { 1119.31102028328, 3112.39622101917, -345.250316177530, 189.383800501452, 763.998451230966, 105.176660890121, 3241.04959557388, 247.177973009491, 0.953618446519386, -0.258327309158681, -0.154530654281614, 0.680722842344000, 0, 0, 0, 1 };
const double CAMERA_MATRIX_11[] = { 1097.26722422431, 3135.90264892856, 145.556791342528, 201.289663827002, -590.488981452968, 74.7230996018302, 3277.94063529018, 214.214368685794, 0.941713191591200, -0.239475514455216, 0.236280712517188, 0.690344379027000, 0, 0, 0, 1 };
const double CAMERA_MATRIX_12[] = { 881.341452231420, 3152.38338361564, 587.183996615913, 207.094794954453, -1846.92277284914, 20.0583947937904, 2772.65847386882, 169.943502800365, 0.774956801062963, -0.226119151790717, 0.590179833053037, 0.695153418505000, 0, 0, 0, 1 };
const double CAMERA_MATRIX_13[] = { 507.286113830152, 3159.10958928477, 906.507852845014, 205.837995859477, -2797.26598953763, -49.7662184968288, 1808.86637482095, 121.695626450840, 0.480960350306278, -0.220469731830845, 0.848569041232862, 0.694353692934000, 0, 0, 0, 1 };
const double CAMERA_MATRIX_14[] = { 37.0363241738984, 3154.96755742156, 1050.65552611546, 197.727363679815, -3284.16321653901, -123.189360564494, 546.146537153530, 77.4594991931220, 0.108403010808193, -0.223462671542792, 0.968664907775209, 0.688077618480000, 0, 0, 0, 1 };
const double CAMERA_MATRIX_15[] = { -451.545179778386, 3140.64311488768, 995.759419643627, 184.105835793500, -3226.99527752196, -188.053818533697, -806.423172995474, 44.5596239111602, -0.281028142252221, -0.234602407754282, 0.930582271491000, 0.677364371223000, 0, 0, 0, 1 };

void loadDinosaurImages(std::vector<gs::StereoImage*> &images)
{
	gs::StereoImage* stereoImage;
	images.clear();

	cv::Mat im0 = cv::imread(DINOSAUR_IMAGE_0, CV_LOAD_IMAGE_COLOR);
	stereoImage = new gs::StereoImage(im0, CAMERA_MATRIX_0);
	images.push_back(stereoImage);

	cv::Mat im1 = cv::imread(DINOSAUR_IMAGE_1, CV_LOAD_IMAGE_COLOR);
	stereoImage = new gs::StereoImage(im1, CAMERA_MATRIX_1);
	images.push_back(stereoImage);

	cv::Mat im2 = cv::imread(DINOSAUR_IMAGE_2, CV_LOAD_IMAGE_COLOR);
	stereoImage = new gs::StereoImage(im2, CAMERA_MATRIX_2);
	images.push_back(stereoImage);

	cv::Mat im3 = cv::imread(DINOSAUR_IMAGE_3, CV_LOAD_IMAGE_COLOR);
	stereoImage = new gs::StereoImage(im3, CAMERA_MATRIX_3);
	images.push_back(stereoImage);

	cv::Mat im4 = cv::imread(DINOSAUR_IMAGE_4, CV_LOAD_IMAGE_COLOR);
	stereoImage = new gs::StereoImage(im4, CAMERA_MATRIX_4);
	images.push_back(stereoImage);

	cv::Mat im5 = cv::imread(DINOSAUR_IMAGE_5, CV_LOAD_IMAGE_COLOR);
	stereoImage = new gs::StereoImage(im5, CAMERA_MATRIX_5);
	images.push_back(stereoImage);

	cv::Mat im6 = cv::imread(DINOSAUR_IMAGE_6, CV_LOAD_IMAGE_COLOR);
	stereoImage = new gs::StereoImage(im6, CAMERA_MATRIX_6);
	images.push_back(stereoImage);

	cv::Mat im7 = cv::imread(DINOSAUR_IMAGE_7, CV_LOAD_IMAGE_COLOR);
	stereoImage = new gs::StereoImage(im7, CAMERA_MATRIX_7);
	images.push_back(stereoImage);

	cv::Mat im8 = cv::imread(DINOSAUR_IMAGE_8, CV_LOAD_IMAGE_COLOR);
	stereoImage = new gs::StereoImage(im8, CAMERA_MATRIX_8);
	images.push_back(stereoImage);

	cv::Mat im9 = cv::imread(DINOSAUR_IMAGE_9, CV_LOAD_IMAGE_COLOR);
	stereoImage = new gs::StereoImage(im9, CAMERA_MATRIX_9);
	images.push_back(stereoImage);

	cv::Mat im10 = cv::imread(DINOSAUR_IMAGE_10, CV_LOAD_IMAGE_COLOR);
	stereoImage = new gs::StereoImage(im10, CAMERA_MATRIX_10);
	images.push_back(stereoImage);

	cv::Mat im11 = cv::imread(DINOSAUR_IMAGE_11, CV_LOAD_IMAGE_COLOR);
	stereoImage = new gs::StereoImage(im11, CAMERA_MATRIX_11);
	images.push_back(stereoImage);

	cv::Mat im12 = cv::imread(DINOSAUR_IMAGE_12, CV_LOAD_IMAGE_COLOR);
	stereoImage = new gs::StereoImage(im12, CAMERA_MATRIX_12);
	images.push_back(stereoImage);

	cv::Mat im13 = cv::imread(DINOSAUR_IMAGE_13, CV_LOAD_IMAGE_COLOR);
	stereoImage = new gs::StereoImage(im13, CAMERA_MATRIX_13);
	images.push_back(stereoImage);

	cv::Mat im14 = cv::imread(DINOSAUR_IMAGE_14, CV_LOAD_IMAGE_COLOR);
	stereoImage = new gs::StereoImage(im14, CAMERA_MATRIX_14);
	images.push_back(stereoImage);

	cv::Mat im15 = cv::imread(DINOSAUR_IMAGE_15, CV_LOAD_IMAGE_COLOR);
	stereoImage = new gs::StereoImage(im15, CAMERA_MATRIX_15);
	images.push_back(stereoImage);
}

int main()
{
	std::vector<gs::StereoImage*> si;
	loadDinosaurImages(si);

	float range[] = { -0.061897, -0.018874, -0.057845, 0.010897, 0.068227, 0.015495 };
	const int rayMarchingSteps = 250;
	const float nccThres = 0.94;

	std::vector<gs::PointCloud*> pointCloud;
	gs::computePointCloud(si, range, rayMarchingSteps, nccThres, pointCloud);
	gs::exportPointCloud(pointCloud, "../exports/pointCloud.bin");

	pointCloud.clear();
	si.clear();
	return 1;
}
```

[1] Goesele, Michael, Brian Curless, and Steven M. Seitz. "Multi-view stereo revisited." Computer Vision and Pattern Recognition, 2006 IEEE Computer Society Conference on. Vol. 2. IEEE, 2006.