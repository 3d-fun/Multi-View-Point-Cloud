#include "PointCloud.h"

gs::PointCloud::PointCloud()
{
	position[0] = 0.0;
	position[1] = 0.0;
	position[2] = 0.0;

	normal[0] = 0.0;
	normal[1] = 0.0;
	normal[2] = 0.0;

	color[0] = 0.0;
	color[1] = 0.0;
	color[2] = 0.0;
}
gs::PointCloud::PointCloud(float* position, float* normal, float* color)
{
	this->position[0] = position[0];
	this->position[1] = position[1];
	this->position[2] = position[2];

	this->normal[0] = normal[0];
	this->normal[1] = normal[1];
	this->normal[2] = normal[2];

	this->color[0] = color[0];
	this->color[1] = color[1];
	this->color[2] = color[2];
}
gs::PointCloud::PointCloud(float px, float py, float pz, float nx, float ny, float nz, float cx, float cy, float cz)
{
	this->position[0] = px;
	this->position[1] = py;
	this->position[2] = pz;

	this->normal[0] = nx;
	this->normal[1] = ny;
	this->normal[2] = nz;

	this->color[0] = cx;
	this->color[1] = cy;
	this->color[2] = cz;
}

gs::Vertex::Vertex()
{
	this->pos[0] = 0.0;
	this->pos[1] = 0.0;
	this->pos[2] = 0.0;
}

gs::Vertex::Vertex(float x, float y, float z)
{
	this->pos[0] = x;
	this->pos[1] = y;
	this->pos[2] = z;
}

gs::Vertex::Vertex(Vertex& v)
{
	this->pos[0] = v.pos[0];
	this->pos[1] = v.pos[1];
	this->pos[2] = v.pos[2];
}

gs::Vertex::~Vertex()
{

}

gs::PointCloud::~PointCloud()
{

}

gs::StereoImage::StereoImage()
{

}

gs::StereoImage::StereoImage(cv::Mat& image, cv::Mat& cameraMatrix)
{
	this->image = image;
	this->cameraMatrix = cameraMatrix;
	this->cameraMatrixInv = this->cameraMatrix.inv();
}

gs::StereoImage::StereoImage(cv::Mat& image, const double* cameraMatrix)
{
	this->image = image;
	this->cameraMatrix = cv::Mat(4, 4, CV_64F, (void*)cameraMatrix);
	this->cameraMatrixInv = this->cameraMatrix.inv();
}


gs::StereoImage::~StereoImage()
{
	this->image.release();
	this->cameraMatrix.release();
}

gs::StereoImage::StereoImage(const StereoImage &stereoImage)
{
	stereoImage.image.copyTo(this->image);
	stereoImage.cameraMatrix.copyTo(this->cameraMatrix);
}

void gs::StereoImage::rotX(float rotation, cv::Mat& result)
{
	float c = cos(rotation);
	float s = sin(rotation);
	result = cv::Mat::zeros(4, 4, CV_64F);
	result.at<double>(0, 0) = 1.0;
	result.at<double>(1, 1) = c;
	result.at<double>(1, 2) = -s;
	result.at<double>(2, 1) = s;
	result.at<double>(2, 2) = c;
	result.at<double>(3, 3) = 1.0;
}
void gs::StereoImage::rotY(float rotation, cv::Mat& result)
{
	float c = cos(rotation);
	float s = sin(rotation);
	result = cv::Mat::zeros(4, 4, CV_64F);
	result.at<double>(0, 0) = c;
	result.at<double>(0, 2) = s;
	result.at<double>(2, 0) = -s;
	result.at<double>(1, 1) = 1.0;
	result.at<double>(2, 2) = c;
	result.at<double>(3, 3) = 1.0;
}
void gs::StereoImage::rotZ(float rotation, cv::Mat& result)
{
	float c = cos(rotation);
	float s = sin(rotation);
	result = cv::Mat::zeros(4, 4, CV_64F);
	result.at<double>(0, 0) = c;
	result.at<double>(0, 1) = -s;
	result.at<double>(1, 0) = s;
	result.at<double>(1, 1) = c;
	result.at<double>(2, 2) = 1.0;
	result.at<double>(3, 3) = 1.0;
}

void gs::StereoImage::computeScreenCoord(cv::Mat& globalPos, cv::Mat& screenCoord)
{
	screenCoord = this->cameraMatrix*globalPos;
	screenCoord.at<double>(0, 0) = screenCoord.at<double>(0, 0) / screenCoord.at<double>(2, 0);
	screenCoord.at<double>(1, 0) = screenCoord.at<double>(1, 0) / screenCoord.at<double>(2, 0);
	screenCoord.at<double>(2, 0) = 1.0;
}

void gs::StereoImage::computeGlobalCoord(cv::Mat& screenCoord, cv::Mat& globalPos)
{
	globalPos = this->cameraMatrixInv*screenCoord;
}

bool gs::StereoImage::isClipped(cv::Mat& screenCoord)
{
	if (screenCoord.at<float>(0, 0) >= this->image.rows || screenCoord.at<float>(0, 0) < 0.0)
		return true;
	if (screenCoord.at<float>(1, 0) >= this->image.cols || screenCoord.at<float>(1, 0) < 0.0)
		return true;
	return false;
}

void gs::StereoImage::sampleImage(int x, int y, int window, unsigned char* sample)
{
	int width = this->image.rows;
	int height = this->image.cols;
	int m = window / 2;
	int startX = x - m;
	int startY = y - m;
	int endX = startX + window;
	int endY = startY + window;

	int sampleIndex = 0;
	for (int sampleX = startX; sampleX < endX; sampleX++)
	{
		for (int sampleY = startY; sampleY < endY; sampleY++)
		{
			if (sampleX >= 0 && sampleX < width && sampleY >= 0 && sampleY < height)
			{
				cv::Vec3b intensity = this->image.at<cv::Vec3b>(sampleX, sampleY);
				sample[sampleIndex] = intensity.val[0];
				sample[sampleIndex + 1] = intensity.val[1];
				sample[sampleIndex + 2] = intensity.val[2];
			}
			else
			{
				sample[sampleIndex] = 0.0;
				sample[sampleIndex + 1] = 0.0;
				sample[sampleIndex + 2] = 0.0;
			}
			sampleIndex = sampleIndex + 3;
		}
	}
}

void gs::normalize(float* a)
{
	float l = length(a);
	if (l == 0)
		return;

	a[0] /= l;
	a[1] /= l;
	a[2] /= l;
}
float gs::length(float* a)
{
	float l = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
	return sqrt(l);
}

void gs::crossProduct(float* a, float* b, float* result)
{
	result[0] = a[1] * b[2] - b[1] * a[2];
	result[1] = -a[0] * b[2] + b[0] * a[2];
	result[2] = a[0] * b[1] - b[0] * a[1];
}

float gs::dotProduct(float* a, float* b)
{
	float r = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
	return r;
}

float gs::normalizedCrossCorrelation(unsigned char* v0, unsigned char* v1, int numSamples)
{
	float num = 0.0;
	float den1 = 0.0;
	float den2 = 0.0;

	float meanV0[] = { 0.0, 0.0, 0.0 };
	float meanV1[] = { 0.0, 0.0, 0.0 };

	for (int i = 0; i < numSamples; i++)
	{
		meanV0[0] += (float)v0[i * 3];
		meanV0[1] += (float)v0[i * 3 + 1];
		meanV0[2] += (float)v0[i * 3 + 2];

		meanV1[0] += (float)v1[i * 3];
		meanV1[1] += (float)v1[i * 3 + 1];
		meanV1[2] += (float)v1[i * 3 + 2];
	}
	meanV0[0] = meanV0[0] / (float)numSamples;
	meanV0[1] = meanV0[1] / (float)numSamples;
	meanV0[2] = meanV0[2] / (float)numSamples;

	meanV1[0] = meanV1[0] / (float)numSamples;
	meanV1[1] = meanV1[1] / (float)numSamples;
	meanV1[2] = meanV1[2] / (float)numSamples;

	for (int i = 0; i < numSamples; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			num += ((float)v0[i * 3 + j] - meanV0[j])*((float)v1[i * 3 + j] - meanV1[j]);
			den1 += pow((float)v0[i * 3 + j] - meanV0[j], 2.0);
			den2 += pow(((float)v1[i * 3 + j] - meanV1[j]), 2.0);
		}
	}
	float den = den1*den2;
	if (den == 0.0)
		return 0.0;
	return num / (sqrt(den));
}

float gs::normalizedDistance(unsigned char* v0, unsigned char* v1, int numSamples)
{
	float sum = 0.0;

	for (int i = 0; i < numSamples; i++)
	{
		sum += pow((float)v0[i * 3] - (float)v1[i * 3], 2.0);
		sum += pow((float)v0[i * 3 + 1] - (float)v1[i * 3 + 1], 2.0);
		sum += pow((float)v0[i * 3 + 2] - (float)v1[i * 3 + 2], 2.0);
	}
	return sum / (float)numSamples;
}

void gs::cameraSubset(int centerCamera, int subsetSize, int numCameras, int* subset)
{
	int n = floor(subsetSize / 2);
	int ssIndex = 0;
	for (int i = -n; i < 0; i++)
	{
		subset[ssIndex] = centerCamera + i;
		if (subset[ssIndex] < 0)
			subset[ssIndex] += numCameras;
		if (subset[ssIndex] >= numCameras)
			subset[ssIndex] += -numCameras;

		ssIndex++;
	}
	for (int i = 0; i <= n; i++)
	{
		subset[ssIndex] = centerCamera + i + 1;
		if (subset[ssIndex] < 0)
			subset[ssIndex] += numCameras;
		if (subset[ssIndex] >= numCameras)
			subset[ssIndex] += -numCameras;

		ssIndex++;
	}
}

void gs::getLocalDepthRange(cv::Mat& cameraMatrix, float* globalRange, double* depthRange)
{
	cv::Mat globalPos(4, 1, CV_64F);
	cv::Mat screenPos(4, 1, CV_64F);

	double maxD;
	double minD;

	globalPos.at<double>(0, 0) = globalRange[0];
	globalPos.at<double>(1, 0) = globalRange[1];
	globalPos.at<double>(2, 0) = globalRange[2];
	globalPos.at<double>(3, 0) = 1.0;
	screenPos = cameraMatrix*globalPos;
	maxD = screenPos.at<double>(2, 0);
	minD = screenPos.at<double>(2, 0);


	globalPos.at<double>(0, 0) = globalRange[0];
	globalPos.at<double>(1, 0) = globalRange[1];
	globalPos.at<double>(2, 0) = globalRange[5];
	globalPos.at<double>(3, 0) = 1.0;
	screenPos = cameraMatrix*globalPos;
	maxD = MAX(screenPos.at<double>(2, 0), maxD);
	minD = MIN(screenPos.at<double>(2, 0), minD);


	globalPos.at<double>(0, 0) = globalRange[0]; //3
	globalPos.at<double>(1, 0) = globalRange[4]; //4
	globalPos.at<double>(2, 0) = globalRange[2]; //5
	globalPos.at<double>(3, 0) = 1.0;
	screenPos = cameraMatrix*globalPos;
	maxD = MAX(screenPos.at<double>(2, 0), maxD);
	minD = MIN(screenPos.at<double>(2, 0), minD);


	globalPos.at<double>(0, 0) = globalRange[0]; //3
	globalPos.at<double>(1, 0) = globalRange[4]; //4
	globalPos.at<double>(2, 0) = globalRange[5]; //5
	globalPos.at<double>(3, 0) = 1.0;
	screenPos = cameraMatrix*globalPos;
	maxD = MAX(screenPos.at<double>(2, 0), maxD);
	minD = MIN(screenPos.at<double>(2, 0), minD);


	globalPos.at<double>(0, 0) = globalRange[3]; //3
	globalPos.at<double>(1, 0) = globalRange[1]; //4
	globalPos.at<double>(2, 0) = globalRange[2]; //5
	globalPos.at<double>(3, 0) = 1.0;
	screenPos = cameraMatrix*globalPos;
	maxD = MAX(screenPos.at<double>(2, 0), maxD);
	minD = MIN(screenPos.at<double>(2, 0), minD);


	globalPos.at<double>(0, 0) = globalRange[3]; //3
	globalPos.at<double>(1, 0) = globalRange[1]; //4
	globalPos.at<double>(2, 0) = globalRange[5]; //5
	globalPos.at<double>(3, 0) = 1.0;
	screenPos = cameraMatrix*globalPos;
	maxD = MAX(screenPos.at<double>(2, 0), maxD);
	minD = MIN(screenPos.at<double>(2, 0), minD);


	globalPos.at<double>(0, 0) = globalRange[3]; //3
	globalPos.at<double>(1, 0) = globalRange[4]; //4
	globalPos.at<double>(2, 0) = globalRange[2]; //5
	globalPos.at<double>(3, 0) = 1.0;
	screenPos = cameraMatrix*globalPos;
	maxD = MAX(screenPos.at<double>(2, 0), maxD);
	minD = MIN(screenPos.at<double>(2, 0), minD);


	globalPos.at<double>(0, 0) = globalRange[3]; //3
	globalPos.at<double>(1, 0) = globalRange[4]; //4
	globalPos.at<double>(2, 0) = globalRange[5]; //5
	globalPos.at<double>(3, 0) = 1.0;
	screenPos = cameraMatrix*globalPos;
	maxD = MAX(screenPos.at<double>(2, 0), maxD);
	minD = MIN(screenPos.at<double>(2, 0), minD);

	depthRange[0] = minD;
	depthRange[1] = maxD;
}

void gs::computeBounds(cv::Mat& points, float* min, float* max)
{
	min[0] = points.at<float>(0, 0);
	min[1] = points.at<float>(0, 1);
	min[2] = points.at<float>(0, 2);

	max[0] = min[0];
	max[1] = min[1];
	max[2] = min[2];

	for (int i = 0; i < points.cols; i++)
	{
		min[0] = MIN(points.at<float>(i, 0), min[0]);
		max[0] = MAX(points.at<float>(i, 0), max[0]);

		min[1] = MIN(points.at<float>(i, 1), min[1]);
		max[1] = MAX(points.at<float>(i, 1), max[1]);

		min[2] = MIN(points.at<float>(i, 2), min[2]);
		max[2] = MAX(points.at<float>(i, 2), max[2]);
	}
}

float gs::localBasis(float* position, cv::Mat* pointCloud, cv::flann::Index* tree, float* basisX, float* basisY, float* basisZ, int numSamples, float* modelCenter)
{
	cv::Mat searchPoint(1, 3, CV_32F);
	searchPoint.at<float>(0, 0) = position[0];
	searchPoint.at<float>(0, 1) = position[1];
	searchPoint.at<float>(0, 2) = position[2];

	cv::Mat ind;
	cv::Mat dist;
	tree->knnSearch(searchPoint, ind, dist, numSamples);


	cv::Mat covariance = cv::Mat::zeros(3, 3, CV_32F);
	cv::Mat y(1, 3, CV_32F);

	cv::Mat sampleCenter = cv::Mat::zeros(1, 3, CV_32F);
	for (int k = 0; k < numSamples; k++)
	{
		int index = ind.at<int>(0, k);

		sampleCenter.at<float>(0, 0) += pointCloud->at<float>(index, 0);
		sampleCenter.at<float>(0, 1) += pointCloud->at<float>(index, 1);
		sampleCenter.at<float>(0, 2) += pointCloud->at<float>(index, 2);
	}
	sampleCenter.at<float>(0, 0) /= (float)numSamples;
	sampleCenter.at<float>(0, 1) /= (float)numSamples;
	sampleCenter.at<float>(0, 2) /= (float)numSamples;

	float dirFromCenter[] = { sampleCenter.at<float>(0, 0) - modelCenter[0], sampleCenter.at<float>(0, 1) - modelCenter[1] , sampleCenter.at<float>(0, 2) - modelCenter[2] };
	normalize(dirFromCenter);

	for (int k = 0; k < numSamples; k++)
	{
		int index = ind.at<int>(0, k);

		y.at<float>(0, 0) = pointCloud->at<float>(index, 0) - sampleCenter.at<float>(0, 0);
		y.at<float>(0, 1) = pointCloud->at<float>(index, 1) - sampleCenter.at<float>(0, 1);
		y.at<float>(0, 2) = pointCloud->at<float>(index, 2) - sampleCenter.at<float>(0, 2);

		cv::Mat yt;
		cv::transpose(y, yt);
		covariance = covariance + yt*y;
	}
	cv::Mat eigenValues;
	cv::Mat eigenVectors;
	cv::eigen(covariance, eigenValues, eigenVectors);

	basisX[0] = eigenVectors.at<float>(0, 0);
	basisX[1] = eigenVectors.at<float>(0, 1);
	basisX[2] = eigenVectors.at<float>(0, 2);

	basisZ[0] = eigenVectors.at<float>(1, 0);
	basisZ[1] = eigenVectors.at<float>(1, 1);
	basisZ[2] = eigenVectors.at<float>(1, 2);

	basisY[0] = eigenVectors.at<float>(2, 0);
	basisY[1] = eigenVectors.at<float>(2, 1);
	basisY[2] = eigenVectors.at<float>(2, 2);
	if (dotProduct(basisY, dirFromCenter) < 0.0)
	{
		basisY[0] = -basisY[0];
		basisY[1] = -basisY[1];
		basisY[2] = -basisY[2];
	}
	return (eigenValues.at<float>(0, 0) / eigenValues.at<float>(2, 0));
}

void gs::computePointCloud(std::vector<gs::StereoImage*>& images, float* globalRange, int rayMarchIterations, float nccThresh, std::vector<PointCloud*>& pointCloud)
{
	int numImages = images.size();

	// init all the buffers.
	cv::Mat screenCoord_i(4, 1, CV_64F);
	cv::Mat screenCoord_j(4, 1, CV_64F);
	cv::Mat globalCoord(4, 1, CV_64F);
	const int M = SAMPLE_WINDOW_SIZE;
	const int LOCAL_CAMERAS = LOCAL_CAMERA_SIZE;
	int sampleX_j, sampleY_j;
	unsigned char windowSample_i[M*M * 3];
	unsigned char windowSample_j[M*M * 3];

	float* validDepths = new float[rayMarchIterations];
	float* depthCorr = new float[rayMarchIterations];
	for (int i = 0; i < rayMarchIterations; i++)
	{
		validDepths[i] = 0.0;
		depthCorr[i] = 0.0;
	}
	int validDepthIndex = 0;
	double d;
	int imageCompareSet[LOCAL_CAMERAS];

	std::vector<Vertex*> vertexList;
	std::vector<Vertex*> colorList;

	for (int imageIndex_i = 0; imageIndex_i < numImages; imageIndex_i++)
	{
		gs::StereoImage* si = images[imageIndex_i];
		int imageWidth = si->image.cols;
		int imageHeight = si->image.rows;

		//compute the depth search range
		double depthRange[2];
		getLocalDepthRange(si->cameraMatrix, globalRange, depthRange);
		double screenDepthEnd = depthRange[1];
		double screenDepthStart = depthRange[0];
		double screenDepthStep = (screenDepthEnd - screenDepthStart) / (double)rayMarchIterations;

		for (int x = 0; x < imageWidth; x = x + 1)
		{
			for (int y = 0; y < imageHeight; y = y + 1)
			{
				// sample the reference image
				si->sampleImage(y, x, M, windowSample_i);
				validDepthIndex = 0;
				
				// compute the sample sum
				int sampleSum = 0;
				for (int i = 0; i < M*M * 3; i++)
				{
					sampleSum += windowSample_i[i];
				}

				// threshold the samples, reject samples that are too dark.
				if (sampleSum >= 6 * M*M)
				{
					for (int i = 0; i < rayMarchIterations; i++)
					{
						depthCorr[i] = 0.0;
					}

					// ray march iterations.
					for (int rayMarchIndex = 0; rayMarchIndex < rayMarchIterations; rayMarchIndex++)
					{
						d = screenDepthStart + rayMarchIndex*screenDepthStep;
						screenCoord_i.at<double>(0, 0) = (double)x*d;
						screenCoord_i.at<double>(1, 0) = (double)y*d;
						screenCoord_i.at<double>(2, 0) = d;
						screenCoord_i.at<double>(3, 0) = 1.0;

						int acceptedNcc = 0;
						//get the set of nearby camears from the refence camera.
						cameraSubset(imageIndex_i, LOCAL_CAMERAS, numImages, imageCompareSet);

						for (int localCameraIndex = 0; localCameraIndex < LOCAL_CAMERAS; localCameraIndex++)
						{
							int imageIndex_j = imageCompareSet[localCameraIndex];
							if (imageIndex_i != imageIndex_j)
							{
								gs::StereoImage* sj = images[imageIndex_j];

								// get the global coordinates raytraced from the reference camera
								si->computeGlobalCoord(screenCoord_i, globalCoord);
								// project the global coordinates onto the comparison camera image plane.
								sj->computeScreenCoord(globalCoord, screenCoord_j);

								if (!sj->isClipped(screenCoord_j))
								{
									sampleX_j = (int)round(screenCoord_j.at<double>(0, 0));
									sampleY_j = (int)round(screenCoord_j.at<double>(1, 0));

									// sample the comparison image
									sj->sampleImage(sampleY_j, sampleX_j, M, windowSample_j);

									// compute the normalized cross correlation.
									float ncc = gs::normalizedCrossCorrelation(windowSample_i, windowSample_j, M*M);

									if (ncc > nccThresh)
									{
										depthCorr[validDepthIndex] += ncc;
										acceptedNcc++;
									}
								}
							}
						}
						depthCorr[validDepthIndex] = depthCorr[validDepthIndex] / (float)acceptedNcc;
						//record all depth sampled that have high correlation between multiple cameras (valid depth).
						if (acceptedNcc >= 2)
						{
							validDepths[validDepthIndex] = d;
							validDepthIndex++;
						}
						else
						{
							depthCorr[validDepthIndex] = 0.0;
						}
					}
				}

				float maxCorrDepth;
				if (validDepthIndex == 0)
				{
					maxCorrDepth = 0.0;
				}
				else
				{
					// search all valid depth samples and find the depth with the highest correlation.
					int maxCorrDepthIndex = 0;
					float maxCorr = depthCorr[0];
					maxCorrDepth = validDepths[0];
					for (int i = 1; i < validDepthIndex; i++)
					{
						if (depthCorr[i] > maxCorr)
						{
							maxCorr = depthCorr[i];
							maxCorrDepthIndex = i;
							maxCorrDepth = validDepths[i];
						}
					}

					screenCoord_i.at<double>(0, 0) = (double)x*maxCorrDepth;
					screenCoord_i.at<double>(1, 0) = (double)y*maxCorrDepth;
					screenCoord_i.at<double>(2, 0) = maxCorrDepth;
					screenCoord_i.at<double>(3, 0) = 1.0;
					si->computeGlobalCoord(screenCoord_i, globalCoord);

					// record the point in the point cloud.
					float pointV[] = { globalCoord.at<double>(0, 0), globalCoord.at<double>(1, 0), globalCoord.at<double>(2, 0) };
					unsigned char colorSample[3];
					si->sampleImage(y, x, 1, colorSample);
					float pointC[] = { ((float)colorSample[0])/ 255.0, ((float)colorSample[1]) / 255.0, ((float)colorSample[2]) / 255.0 };

					Vertex* v = new Vertex(pointV[0], pointV[1], pointV[2]);
					Vertex* c = new Vertex(pointC[0], pointC[1], pointC[2]);
					vertexList.push_back(v);
					colorList.push_back(c);
				}
			}
		}
	}

	cv::Mat cloud(vertexList.size(), 3, CV_32F);
	for (int i = 0; i < vertexList.size(); i++)
	{
		Vertex* v = vertexList[i];
		cloud.at<float>(i, 0) = v->pos[0];
		cloud.at<float>(i, 1) = v->pos[1];
		cloud.at<float>(i, 2) = v->pos[2];
	}
	cv::flann::Index* tree = new cv::flann::Index(cloud, cv::flann::KDTreeIndexParams(), cvflann::FLANN_DIST_EUCLIDEAN);

	float boundsMin[3];
	float boundsMax[3];
	computeBounds(cloud, boundsMin, boundsMax);
	float modelCenter[3];
	modelCenter[0] = (boundsMin[0] + boundsMax[0])*0.5;
	modelCenter[1] = (boundsMin[1] + boundsMax[1])*0.5;
	modelCenter[2] = (boundsMin[2] + boundsMax[2])*0.5;

	pointCloud.clear();
	float basisX[3];
	float basisY[3];
	float basisZ[3];
	for (int i = 0; i < vertexList.size(); i++)
	{
		Vertex* v = vertexList[i];
		Vertex* c = colorList[i];
		float conditionNumber = localBasis(v->pos, &cloud, tree, basisX, basisY, basisZ, 40, modelCenter);
		if (conditionNumber > 5.0)
		{
			PointCloud* pc = new PointCloud(v->pos, basisY, c->pos);
			pointCloud.push_back(pc);
		}
	}

	vertexList.clear();
	colorList.clear();

	cloud.release();
	tree->release();

	screenCoord_i.release();
	screenCoord_j.release();
	globalCoord.release();
	delete[] validDepths;
	delete[] depthCorr;
}

void gs::exportPointCloud(std::vector<PointCloud*>& pointCloud, const char* filePath)
{
	std::fstream file;
	file.open(filePath, std::fstream::out | std::fstream::binary);

	int numPoints = pointCloud.size();
	file.write((const char*)&numPoints, 4);

	for (int i = 0; i < numPoints; i++)
	{
		PointCloud* pc = pointCloud[i];
		float v;
		/*write positions*/
		v = pc->position[0];
		file.write((const char*)&v, 4);

		v = pc->position[1];
		file.write((const char*)&v, 4);

		v = pc->position[2];
		file.write((const char*)&v, 4);


		/*write normals*/
		v = pc->normal[0];
		file.write((const char*)&v, 4);

		v = pc->normal[1];
		file.write((const char*)&v, 4);

		v = pc->normal[2];
		file.write((const char*)&v, 4);


		/*write colors*/
		v = pc->color[0];
		file.write((const char*)&v, 4);

		v = pc->color[1];
		file.write((const char*)&v, 4);

		v = pc->color[2];
		file.write((const char*)&v, 4);
	}
	file.close();
}