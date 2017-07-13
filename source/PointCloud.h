/**
PointCloud.h
Computes a point cloud from a series of stereoscopic images.
This algorithm utilizes ray marching and normalized cross correlation to compute depth matching between camera views.

The point cloud contains a set of:
	points
	normals
	colors

Hyperparameters:
	window size: image widow sampling size.
	local camera size: local subset of views in which to compare with the reference.
	minimum condition number: remove points in which the condition number of the covariance matrix is less than a minimum.

@author Greg Smith, 2017
*/

#pragma once
#include <opencv2\opencv.hpp>
#include <fstream>

#define SAMPLE_WINDOW_SIZE 5
#define LOCAL_CAMERA_SIZE 4
#define MIN_CONDITION_NUMBER 5.0

namespace gs
{
	/*SteroImage
	
	Class containing an image with its corresponding camera matrix.

	Mat image : 8 bit 3 color image.
	Mat cameraMatrix. 
	Mat cameraMatrixInv: inverse camera matrix.
	*/
	class StereoImage
	{
	public:

		/*constructors, desctructor, copy*/
		StereoImage();
		StereoImage(cv::Mat& image, cv::Mat& cameraMatrix);
		StereoImage(cv::Mat& image, const double* cameraMatrix);
		StereoImage(const StereoImage &stereoImage);
		virtual ~StereoImage();

		cv::Mat image;
		cv::Mat cameraMatrix;
		cv::Mat cameraMatrixInv;

		/*
		computeScreenCoord 
		Projects a global position to a local screen coordinate.
		
		@param Mat& globalPosition
		@param Mat& screenCoord (return)
		*/
		void computeScreenCoord(cv::Mat& globalPos, cv::Mat& screenCoord);

		/*
		computeGlobalCoord
		computes the global coordinates from local screen coordinates with a z-depth.

		@param Mat& screenCoord
		@param Mat& globalPos (return)
		*/
		void computeGlobalCoord(cv::Mat& screenCoord, cv::Mat& globalPos);

		/*
		isClipped
		determines if a screen coordinate is outside the image viewing plane.

		@param Mat& screenCoord
		*/
		bool isClipped(cv::Mat& screenCoord);

		/*
		sampleImage
		samples a window of pixels from 'image' centered at (x,y)

		@param int x: center pixel x.
		@param int y: center pixel y.
		@param int window: window size.
		@param uchar* sample: buffer for image sample. must be size (3*3*window).
		*/
		void sampleImage(int x, int y, int window, unsigned char* sample);

	private:
		void rotX(float rotation, cv::Mat& result);
		void rotY(float rotation, cv::Mat& result);
		void rotZ(float rotation, cv::Mat& result);
	};

	class Vertex
	{
	public:
		Vertex();
		Vertex(float x, float y, float z);
		Vertex(Vertex& v);
		virtual ~Vertex();

		float pos[3];
	};

	class PointCloud
	{
	public:
		PointCloud();
		PointCloud(float* position, float* normal, float* color);
		PointCloud(float px, float py, float pz, float nx, float ny, float nz, float cx, float cy, float cz);
		virtual ~PointCloud();

		float position[3];
		float normal[3];
		float color[3];
	};

	/*
	computePointCloud
	computes a point cloud from a set of stereoscopic images. returned in 'pointCloud'.

	@param vector<SteroImage*> stereoImages
	@param float[6] globalDepthRange: range in which the points are searched
	@param int rayMarchIterations: number of 
	*/
	void computePointCloud(std::vector<gs::StereoImage*>& images, float* globalDepthRange, int rayMarchIterations, float nccThresh, std::vector<PointCloud*>& pointCloud);	
	
	/*
	float normalizedCroddCorrelation
	computes a point cloud from a set of stereoscopic images. returned in 'pointCloud'.

	@param unsigned char[numSamples*3] v0: input vector 0, total vector size : numSamples*3
	@param unsigned char[numSamples*3] v1: input vector 1, total vector size : numSamples*3
	@param int numSamples: number of color samples in a vector 

	@return ncc [-1,1] 
	*/
	float normalizedCrossCorrelation(unsigned char* v0, unsigned char* v1, int numSamples);
	float normalizedDistance(unsigned char* v0, unsigned char* v1, int numSamples);
	
	/*
	cameraSubset
	returns the [subsetSize] set of cameras closest to the reference camera.

	@param centerCamera: reference camera index.
	@param subsetSize: local camera subsetSize.
	@param numCameras: total number of cameras in the scene.
	@param int[subsetSize]: subset return array. 
	*/
	void cameraSubset(int centerCamera, int subsetSize, int numCameras, int* subset);
	
	/*
	getLocalDepthRange
	computes the local depth range from a camera from global coordinates and places it in depthRange.

	@param Mat& cameraMatrix: camera transform matrix
	@param float[6] globalRange
	@param double[2] depthRange
	*/
	void getLocalDepthRange(cv::Mat& cameraMatrix, float* globalRange, double* depthRange);
	
	/*
	computeBounds
	given an array of points, find the minimum and maximum points and return them in min and max respectively

	@param Mat& points: matrix of points of dimensions ( numPoints, 3)
	@param float[3] min
	@param float[3] max
	*/
	void computeBounds(cv::Mat& points, float* min, float* max);
	

	/*
	float localBasis
	estimates the local basis of a point 'position'. We compute the eigen vectors of the covariance matrix.
	result is returned in basisX[3], basisY[3], basisZ[3].

	@param float[3] position:
	@param Mat* pointCloud: point cloud matrix of dimensions (numPoints, 3)
	@param flann::Index* tree: binary tree used for searching
	@param float[3] basisX
	@param float[3] basisY
	@param float[3] basisZ
	@param numSamples: number of samples used to estimate the covariance matrix
	@param float[3] modelCenter: center position of the model used to orient normals.

	@return float: condition number (largest singular value / smallest singular value).
	*/
	float localBasis(float* position, cv::Mat* pointCloud, cv::flann::Index* tree, float* basisX, float* basisY, float* basisZ, int numSamples, float* modelCenter);
	
	void normalize(float* a);
	void crossProduct(float* a, float* b, float* result);
	float dotProduct(float* a, float* b);
	float length(float* a);

	void exportPointCloud(std::vector<PointCloud*>& pointCloud, const char* filePath);
}