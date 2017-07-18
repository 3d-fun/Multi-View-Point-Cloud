#include "PointCloudKernel.h"

gs::PointCloudKernel::PointCloudKernel(ClContext* context, std::vector<gs::StereoImage*> &stereoImages, float* searchRange, int rayMarchingIters, float nccThresh, std::vector<gs::PointCloud*> &pointCloud) :
	ClKernel(POINT_CLOUD_KERNEL_SOURCE, context, createPointCloudKernelDefines(stereoImages, rayMarchingIters, nccThresh))
{
	__rayMarchIterations = rayMarchingIters;
	//__stereoImages = stereoImages;
	for (int i = 0; i < stereoImages.size(); i++)
	{
		__stereoImages.push_back(stereoImages[i]);
	}
	__nccThresh = nccThresh;

	 __numValidDepthSize = stereoImages[0]->image.rows*stereoImages[0]->image.cols;
	__validDepthSize = stereoImages[0]->image.rows*stereoImages[0]->image.cols;
	__imageBufferSize = stereoImages[0]->image.rows*stereoImages[0]->image.cols*stereoImages.size()*4;
	__numThreads = stereoImages[0]->image.rows*stereoImages[0]->image.cols;

	__stereoImagesHost = nullptr;
	__transformMatrixHost = nullptr;
	__transformMatrixInvHost = nullptr;
	__depthSearchRangeHost = nullptr;
	__numValidDepthsHost = nullptr;
	__validDepthsHost = nullptr;

	int stereoImageBufferSize = 0;
	for (int i = 0; i < stereoImages.size(); i++)
	{
		gs::StereoImage* si = stereoImages[i];
		stereoImageBufferSize += si->image.rows*si->image.cols*4;
	}
	__stereoImagesHost = new unsigned char[stereoImageBufferSize];
	
	int siIndex = 0;
	for (int i = 0; i < stereoImages.size(); i++)
	{
		gs::StereoImage* si = stereoImages[i];
		for (int x = 0; x < si->image.rows; x++)
		{
			for (int y = 0; y < si->image.cols; y++)
			{
				cv::Vec3b sample = si->image.at<cv::Vec3b>(x, y);
				__stereoImagesHost[siIndex * 4] = sample[0];
				__stereoImagesHost[siIndex * 4 + 1] = sample[1];
				__stereoImagesHost[siIndex * 4 + 2] = sample[2];
				__stereoImagesHost[siIndex * 4 + 3] = 1;
				siIndex++;
			}
		}
	}

	__transformMatrixHost = new float[stereoImages.size() * 16];
	int camIndex = 0;
	for (int i = 0; i < stereoImages.size(); i++)
	{
		camIndex = 0;
		for (int x = 0; x < 4; x++)
		{
			for (int y = 0; y < 4; y++)
			{
				__transformMatrixHost[i * 16 + camIndex] = (float)stereoImages[i]->cameraMatrix.at<double>(x, y);
				camIndex++;
			}
		}
	}


	__transformMatrixInvHost = new float[stereoImages.size() * 16];
	for (int i = 0; i < stereoImages.size(); i++)
	{
		camIndex = 0;
		for (int x = 0; x < 4; x++)
		{
			for (int y = 0; y < 4; y++)
			{
				__transformMatrixInvHost[i * 16 + camIndex] = (float)stereoImages[i]->cameraMatrixInv.at<double>(x, y);
				camIndex++;
			}
		}
	}

	float depthRange[2];
	size_t searchRangeSize = stereoImages.size() * 2;
	__depthSearchRangeHost = new float[searchRangeSize];
	for (int i = 0; i < stereoImages.size(); i++)
	{
		gs::StereoImage* si = stereoImages[i];
		getLocalDepthRange(si->cameraMatrix, searchRange, depthRange);
		__depthSearchRangeHost[i * 2] = depthRange[0];
		__depthSearchRangeHost[i * 2 + 1] = depthRange[1];
	}

	__numValidDepthsHost = new int[__validDepthSize];
	__validDepthsHost = new float[__validDepthSize];
	__depthCorrelationHost = new float[__validDepthSize];
	for (int i = 0; i < __numValidDepthSize; i++)
	{
		__numValidDepthsHost[i] = 0;
		__validDepthsHost[i] = 0.0;
		__depthCorrelationHost[i] = 0.0;
	}

	createKernel("computeDepth");

	cl_int ret;
	
	cl_image_format imageFormat;
	imageFormat.image_channel_data_type = CL_UNSIGNED_INT8;
	imageFormat.image_channel_order = CL_RGBA;

	cl_image_desc imageDesc;
	imageDesc.image_type = CL_MEM_OBJECT_IMAGE3D;
	imageDesc.image_width = (size_t)stereoImages[0]->image.cols;
	imageDesc.image_height = (size_t)stereoImages[0]->image.rows;
	imageDesc.image_depth = (size_t)stereoImages.size();
	imageDesc.image_array_size = (size_t)stereoImages.size();
	imageDesc.image_array_size = 0;
	imageDesc.image_row_pitch = 0;
	imageDesc.image_slice_pitch = 0;
	imageDesc.num_mip_levels = 0;
	imageDesc.num_samples = 0;
	imageDesc.buffer = __stereoImagesClient;
	__stereoImagesClient = clCreateImage(__context->context, CL_MEM_READ_ONLY, &imageFormat, &imageDesc, __stereoImagesHost, &ret);

	size_t rowPitchSize;
	clGetImageInfo(__stereoImagesClient, CL_IMAGE_ROW_PITCH, sizeof(size_t), &rowPitchSize, NULL);

	size_t sliceSize;
	clGetImageInfo(__stereoImagesClient, CL_IMAGE_SLICE_PITCH, sizeof(size_t), &sliceSize, NULL);

	size_t origin[] = { 0.0,0.0,0.0 };
	size_t region[] = { (size_t)stereoImages[0]->image.cols, (size_t)stereoImages[0]->image.rows, (size_t)stereoImages.size() };
	ret = clEnqueueWriteImage(__context->commandQueue, __stereoImagesClient, CL_TRUE, origin, region, rowPitchSize, sliceSize, __stereoImagesHost, 0, NULL, NULL);
	clFinish(__context->commandQueue);

	//stereoImages.size() * 16
	__transformMatrixClient = clCreateBuffer(this->__context->context, CL_MEM_READ_WRITE, stereoImages.size() * 16*sizeof(float), NULL, &ret);
	ret = clEnqueueWriteBuffer(__context->commandQueue, __transformMatrixClient, CL_TRUE, 0, stereoImages.size() * 16 * sizeof(float), __transformMatrixHost, 0, NULL, NULL);
	clFinish(__context->commandQueue);

	__transformMatrixInvClient = clCreateBuffer(this->__context->context, CL_MEM_READ_WRITE, stereoImages.size() * 16 * sizeof(float), NULL, &ret);
	ret = clEnqueueWriteBuffer(__context->commandQueue, __transformMatrixInvClient, CL_TRUE, 0, stereoImages.size() * 16 * sizeof(float), __transformMatrixInvHost, 0, NULL, NULL);
	clFinish(__context->commandQueue);

	__depthSearchRangeClient = clCreateBuffer(this->__context->context, CL_MEM_READ_WRITE, searchRangeSize * sizeof(float), NULL, &ret);
	ret = clEnqueueWriteBuffer(__context->commandQueue, __depthSearchRangeClient, CL_TRUE, 0, searchRangeSize * sizeof(float), __depthSearchRangeHost, 0, NULL, NULL);
	clFinish(__context->commandQueue);

	__numValidDepthsClient = clCreateBuffer(this->__context->context, CL_MEM_READ_WRITE, __validDepthSize * sizeof(int), NULL, &ret);
	ret = clEnqueueWriteBuffer(__context->commandQueue, __numValidDepthsClient, CL_TRUE, 0, __validDepthSize * sizeof(int), __numValidDepthsHost, 0, NULL, NULL);
	clFinish(__context->commandQueue);

	__validDepthsClient = clCreateBuffer(this->__context->context, CL_MEM_READ_WRITE, __validDepthSize * sizeof(float), NULL, &ret);
	ret = clEnqueueWriteBuffer(__context->commandQueue, __validDepthsClient, CL_TRUE, 0, __validDepthSize * sizeof(float), __validDepthsHost, 0, NULL, NULL);
	clFinish(__context->commandQueue);

	__depthCorrelationClient = clCreateBuffer(this->__context->context, CL_MEM_READ_WRITE, __validDepthSize * sizeof(float), NULL, &ret);
	ret = clEnqueueWriteBuffer(__context->commandQueue, __depthCorrelationClient, CL_TRUE, 0, __validDepthSize * sizeof(float), __depthCorrelationHost, 0, NULL, NULL);
	clFinish(__context->commandQueue);

	cl_int numSampleTemp = 0;

	/* ACTIVATION LAYER 1*/
	addKernelArg(0, 0, sizeof(cl_mem), (void*)&__stereoImagesClient);
	addKernelArg(0, 1, sizeof(cl_mem), (void*)&__transformMatrixClient);
	addKernelArg(0, 2, sizeof(cl_mem), (void*)&__transformMatrixInvClient);
	addKernelArg(0, 3, sizeof(cl_mem), (void*)&__depthSearchRangeClient);
	addKernelArg(0, 4, sizeof(cl_mem), (void*)&__numValidDepthsClient);
	addKernelArg(0, 5, sizeof(cl_mem), (void*)&__validDepthsClient);
	addKernelArg(0, 6, sizeof(cl_mem), (void*)&__depthCorrelationClient);
	cl_int depthIndex = 0;
	cl_int imageReferenceIndex = 0;
	addKernelArg(0, 7, sizeof(cl_int), (void*)&imageReferenceIndex);
	addKernelArg(0, 8, sizeof(cl_int), (void*)&depthIndex);

	computePointCloud(pointCloud);
}

gs::PointCloudKernel::~PointCloudKernel()
{
	delete[] __stereoImagesHost;
	delete[] __transformMatrixHost;
	delete[] __transformMatrixInvHost;
	delete[] __depthSearchRangeHost;
	delete[] __numValidDepthsHost;
	delete[] __validDepthsHost;
	delete[] __depthCorrelationHost;

	clReleaseMemObject(__stereoImagesClient);
	clReleaseMemObject(__transformMatrixClient);
	clReleaseMemObject(__transformMatrixInvClient);
	clReleaseMemObject(__depthSearchRangeClient);
	clReleaseMemObject(__numValidDepthsClient);
	clReleaseMemObject(__depthCorrelationClient);
}

std::vector<std::string> gs::PointCloudKernel::createPointCloudKernelDefines(std::vector<gs::StereoImage*> &stereoImages, int rayMarchIters, float nccThresh)
{
	std::vector<std::string> def;

	std::string numImagesString("#define NUM_STEREO_IMAGES ");
	numImagesString = numImagesString + std::to_string(stereoImages.size());
	numImagesString = numImagesString + std::string("\n");
	def.push_back(numImagesString);

	std::string rayMarchItersString("#define RAY_MARCH_ITERS ");
	rayMarchItersString = rayMarchItersString + std::to_string(rayMarchIters);
	rayMarchItersString = rayMarchItersString + std::string("\n");
	def.push_back(rayMarchItersString);

	std::string nccThreshString("#define NCC_THRESHOLD ");
	nccThreshString = nccThreshString + std::to_string(nccThresh);
	nccThreshString = nccThreshString + std::string("\n");
	def.push_back(nccThreshString);

	return def;
}

void gs::PointCloudKernel::computePointCloud(std::vector<gs::PointCloud*> &pointCloud)
{
	pointCloud.clear();
	cl_int depthIndex = 0;
	cl_int imageReferenceIndex = 0;


	int numImages = this->__stereoImages.size();

	for (int imageReferenceIndex = 0; imageReferenceIndex < numImages; imageReferenceIndex++)
	{
		clearValidDepths();
		clearNumValidDepths();
		clearDepthCorrelation();

		addKernelArg(0, 7, sizeof(cl_int), (void*)&imageReferenceIndex);
		for (int i = 0; i < __rayMarchIterations; i++)
		{
			addKernelArg(0, 8, sizeof(cl_int), (void*)&i);
			enqueueKernel(0, __numThreads, 1);
		}

		readValidDepths();
		readNumValidDepths();
		gs::StereoImage* si = __stereoImages[imageReferenceIndex];
		cv::Mat screenPos(4, 1, CV_64F);
		cv::Mat globalPos(4, 1, CV_64F);
		
		float normal[] = { 0.0, 0.0, 0.0 };
		float position[3];
		float color[3];

		int imageIndex = 0;
		for (int x = 0; x < si->image.cols; x++)
		{
			for (int y = 0; y < si->image.rows; y++)
			{
				int validDepths = __numValidDepthsHost[imageIndex];
				if (validDepths > 0)
				{
					float dpc = __validDepthsHost[imageIndex];
					float xpc = ((float)x)*dpc;
					float ypc = ((float)y)*dpc;

					screenPos.at<double>(0, 0) = (double)xpc;
					screenPos.at<double>(1, 0) = (double)ypc;
					screenPos.at<double>(2, 0) = (double)dpc;
					screenPos.at<double>(3, 0) = 1.0;
					globalPos = si->cameraMatrixInv*screenPos;

					position[0] = (float)globalPos.at<double>(0, 0);
					position[1] = (float)globalPos.at<double>(1, 0);
					position[2] = (float)globalPos.at<double>(2, 0);

					cv::Vec3b imageSample;
					imageSample = si->image.at<cv::Vec3b>(y, x);
					color[0] = (float)imageSample[0] / 255.0;
					color[1] = (float)imageSample[1] / 255.0;
					color[2] = (float)imageSample[2] / 255.0;

					gs::PointCloud* pc = new gs::PointCloud(position, normal, color );
					pointCloud.push_back(pc);
				}
				imageIndex++;
			}
		}
	}

	computePointCloudNormals(pointCloud);
}

void gs::PointCloudKernel::readNumValidDepths()
{
	cl_int ret;
	ret = clEnqueueReadBuffer(__context->commandQueue, __numValidDepthsClient, CL_TRUE, 0, __numValidDepthSize * sizeof(int), (void*)__numValidDepthsHost, 0, NULL, NULL);
	clFinish(__context->commandQueue);
}
void gs::PointCloudKernel::readValidDepths()
{
	cl_int ret;
	ret = clEnqueueReadBuffer(__context->commandQueue, __validDepthsClient, CL_TRUE, 0, __validDepthSize * sizeof(float), (void*)__validDepthsHost, 0, NULL, NULL);
	clFinish(__context->commandQueue);
}
void gs::PointCloudKernel::readDepthCorrelation()
{
	cl_int ret;
	ret = clEnqueueReadBuffer(__context->commandQueue, __depthCorrelationClient, CL_TRUE, 0, __validDepthSize * sizeof(float), (void*)__depthCorrelationHost, 0, NULL, NULL);
	clFinish(__context->commandQueue);
}


void gs::PointCloudKernel::clearValidDepths()
{
	for (int i = 0; i < __validDepthSize; i++)
	{
		__validDepthsHost[i] = 0.0;
	}
	clEnqueueWriteBuffer(__context->commandQueue, __validDepthsClient, CL_TRUE, 0, __validDepthSize * sizeof(float), __validDepthsHost, 0, NULL, NULL);
	clFinish(__context->commandQueue);
}
void gs::PointCloudKernel::clearNumValidDepths()
{
	for (int i = 0; i < __numValidDepthSize; i++)
	{
		__numValidDepthsHost[i] = 0;
	}
	clEnqueueWriteBuffer(__context->commandQueue, __numValidDepthsClient, CL_TRUE, 0, __numValidDepthSize * sizeof(int), __numValidDepthsHost, 0, NULL, NULL);
	clFinish(__context->commandQueue);
}
void gs::PointCloudKernel::clearDepthCorrelation()
{
	for (int i = 0; i < __numValidDepthSize; i++)
	{
		__depthCorrelationHost[i] = 0;
	}
	clEnqueueReadBuffer(__context->commandQueue, __depthCorrelationClient, CL_TRUE, 0, __validDepthSize * sizeof(float), (void*)__depthCorrelationHost, 0, NULL, NULL);
	clFinish(__context->commandQueue);
}

void gs::PointCloudKernel::getLocalDepthRange(cv::Mat& cameraMatrix, float* globalRange, float* depthRange)
{
	cv::Mat globalPos(4, 1, CV_64F);
	cv::Mat screenPos(4, 1, CV_64F);

	float maxD;
	float minD;

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