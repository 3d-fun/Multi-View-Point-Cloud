#pragma once

/**
PointCloudKernel.h
Computes a point cloud from a series of stereoscopic images.
This algorithm utilizes ray marching and normalized cross correlation to compute depth matching between camera views.
Computation is done in parallel using openCL.

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

#include "ClKernel.h"
#include "PointCloud.h"

#define POINT_CLOUD_KERNEL_SOURCE "PointCloud.cl"

/*
class PointCloudKernel
computes a point cloud from a set of stereo images.
PointCloudKernel extends ClKernel where point cloud depth computations are done in parallel with the kernel located at: POINT_CLOUD_KERNEL_SOURCE 

constructor:
@param ClContext* context: if no context exists simple create a new one (new ClContext()).
@param std::vector<gs::StereoImage*> &stereoImages: list of stereo images with a camera matrix.
@param float* searchRange : global range of the object to be searched.
@param rayMarchingIters : number of iterations used
@param nccThresh: normalized cross correlation threshold
@param std::vector<gs::PointCloud*> &pointCloud : returned vector of points in the Point Cloud.

*/
namespace gs
{
	class PointCloudKernel : public ClKernel
	{
	public:
		PointCloudKernel(ClContext* context, std::vector<gs::StereoImage*> &stereoImages, float* searchRange, int rayMarchingIters, float nccThresh, std::vector<gs::PointCloud*> &pointCloud);
		virtual ~PointCloudKernel();

	private:
		int __rayMarchIterations;
		std::vector<gs::StereoImage*> __stereoImages;
		float __nccThresh;

		size_t __numValidDepthSize;
		size_t __validDepthSize;
		size_t __imageBufferSize;
		size_t __numThreads;

		unsigned char* __stereoImagesHost;
		float* __transformMatrixHost;
		float* __transformMatrixInvHost;
		float* __depthSearchRangeHost;
		int* __numValidDepthsHost;
		float* __validDepthsHost;
		float* __depthCorrelationHost;

		/*client memory buffers*/
		cl_mem __stereoImagesClient;
		cl_mem __transformMatrixClient;
		cl_mem __transformMatrixInvClient;
		cl_mem __depthSearchRangeClient;
		cl_mem __numValidDepthsClient;
		cl_mem __validDepthsClient;
		cl_mem __depthCorrelationClient;

		/*
		function createPointCloudKernelDefines
		creates a string in which prepends constants to the kernel source
		*/
		std::vector<std::string> createPointCloudKernelDefines(std::vector<gs::StereoImage*> &stereoImages, int rayMarchIters, float nccThresh);

		/*
		function computePointCloud
		computes a vector of point, done in parallel.

		@param std::vector<gs::PointCloud*> &pointCloud : return vector of points in the point cloud
		*/
		void computePointCloud(std::vector<gs::PointCloud*> &pointCloud);

		/*
		function getLocalDepthRange
		computes the local depth range given a global object range. This range is returned in 'depthRange'.

		@param cv::Mat& cameraMatrix : camrea matrix matrix
		@param float* globalRange: range of the object in global coordinates globalRange[] = {minX, minY, minZ, maxX, maxY, maxZ}
		@param float depthRange[2] : local depth range depth
		*/
		void getLocalDepthRange(cv::Mat& cameraMatrix, float* globalRange, float* depthRange);

		/*
		buffer reading routines
		*/
		void readValidDepths();
		void readNumValidDepths();
		void readDepthCorrelation();

		/*
		buffer clearing routines
		*/
		void clearValidDepths();
		void clearNumValidDepths();
		void clearDepthCorrelation();
	};
}