#include "ClContext.h"

ClContext::ClContext()
{
	deviceId = NULL;
	platformId = NULL;
	context = NULL;
	commandQueue = NULL;

	createContext();
	createQueue();
}
ClContext::~ClContext()
{
	cl_int ret;
	ret = clReleaseCommandQueue(commandQueue);
	ret = clReleaseContext(context);
}

void ClContext::createContext()
{
	cl_int ret;
	ret = clGetPlatformIDs(2, &platformId, &retNumPlatforms);

	switch (ret)
	{
		case CL_SUCCESS:
		{
			printf("%i platform(s) found: \n", retNumPlatforms);
			printPlatformInfo(platformId,retNumPlatforms);
			break;
		}
		case CL_INVALID_VALUE:
		{
			printf("ERROR, could not find valid platforms.\n");
			return;
		}
		default:
			break;
	}

	ret = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &deviceId, &retNumDevices);
	switch (ret)
	{
		case CL_SUCCESS:
		{
			printf("Success, device found. \n");
			printDeviceInfo(deviceId);
			break;
		}

		case CL_INVALID_PLATFORM:
		{
			printf("ERROR, platform is not a valid platform.\n");
			break;
		}

		case CL_INVALID_DEVICE_TYPE:
		{
			printf("ERROR, num_entries is equal to zero and device_type is not NULL or both num_devices and device_type are NULL \n");
			break;
		}
		
		case CL_INVALID_VALUE:
		{
			printf("ERROR, no OpenCL devices that matched device_type were found \n");
			break;
		}

		case CL_DEVICE_NOT_FOUND:
		{
			printf("ERROR, device not found \n");
			break;
		}
	}

	context = clCreateContext(NULL, 1, &deviceId, NULL, NULL, &ret);
}
void ClContext::createQueue()
{
	cl_int ret;
	commandQueue = clCreateCommandQueue(context, deviceId, 0, &ret);
}

void ClContext::printPlatformInfo(cl_platform_id platformId, cl_uint retNumPlatforms)
{
	cl_int retPlat;
	size_t returnStringSize;
	char stringRet[128];

	for (int i = 0; i < retNumPlatforms; i++)
	{
		retPlat = clGetPlatformInfo(platformId, CL_PLATFORM_PROFILE, 128, (void*)stringRet, &returnStringSize);
		printf(stringRet);
		printf("\n");

		retPlat = clGetPlatformInfo(platformId, CL_PLATFORM_NAME, 128, (void*)stringRet, &returnStringSize);
		printf(stringRet);
		printf("\n");

		retPlat = clGetPlatformInfo(platformId, CL_PLATFORM_VENDOR, 128, (void*)stringRet, &returnStringSize);
		printf(stringRet);
		printf("\n");

		retPlat = clGetPlatformInfo(platformId, CL_PLATFORM_EXTENSIONS, 128, (void*)stringRet, &returnStringSize);
		printf(stringRet);
		printf("\n");
	}
}

void ClContext::printDeviceInfo(cl_device_id deviceId)
{
	size_t returnedSize;

	cl_int addressBits;
	cl_ulong cacheSize;
	cl_device_mem_cache_type cacheType;
	cl_uint cacheLineSize;
	cl_ulong globalMemSize;

	char deviceName[128];
	char deviceVendor[128];
	char deviceVersion[128];

	clGetDeviceInfo(deviceId, CL_DEVICE_NAME, 128, &deviceName, &returnedSize);
	printf(deviceName);
	printf("\n");

	clGetDeviceInfo(deviceId, CL_DEVICE_VENDOR, 128, &deviceVendor, &returnedSize);
	printf(deviceVendor);
	printf("\n");

	clGetDeviceInfo(deviceId, CL_DEVICE_VERSION, 128, &deviceVersion, &returnedSize);
	printf(deviceVersion);
	printf("\n");


	clGetDeviceInfo(deviceId, CL_DEVICE_ADDRESS_BITS, sizeof(cl_int), &addressBits, &returnedSize);
	printf("address bits: %i \n", addressBits);

	clGetDeviceInfo(deviceId, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &cacheSize, &returnedSize);
	printf("Size of global memory cache in bytes: %i \n", cacheSize);
	
	clGetDeviceInfo(deviceId, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cl_uint), &cacheLineSize, &returnedSize);
	printf("Size of global memory cache line in bytes: %u \n", cacheLineSize);

	clGetDeviceInfo(deviceId, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &globalMemSize, &returnedSize);
	printf("Size of global device memory in bytes: %X \n", globalMemSize);


	char extensions[128];
	clGetDeviceInfo(deviceId, CL_DEVICE_EXTENSIONS, 128, &extensions, &returnedSize);
	printf("extensions: \n");
	printf(extensions);

	cl_bool imageSupport;
	clGetDeviceInfo(deviceId, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &imageSupport, &returnedSize);
	if (imageSupport == CL_FALSE)
		printf("images are not supported \n");
	else if (imageSupport == CL_TRUE)
	{
		printf("images are supported \n");

		size_t image2DMaxHeight, image2DMaxwidth;
		clGetDeviceInfo(deviceId, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &image2DMaxwidth, &returnedSize);
		printf("Max width of 2D image in pixels: %u \n", image2DMaxwidth);
		clGetDeviceInfo(deviceId, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &image2DMaxHeight, &returnedSize);
		printf("Max height of 2D image in pixels: %u \n", image2DMaxHeight);

		size_t image3DMaxHeight, image3DMaxwidth, image3DMaxDepth;
		clGetDeviceInfo(deviceId, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_t), &image3DMaxwidth, &returnedSize);
		printf("Max width of 3D image in pixels: %u \n", image3DMaxwidth);
		clGetDeviceInfo(deviceId, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(size_t), &image3DMaxHeight, &returnedSize);
		printf("Max height of 3D image in pixels: %u \n", image3DMaxHeight);
		clGetDeviceInfo(deviceId, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(size_t), &image3DMaxDepth, &returnedSize);
		printf("Max depth of 3D image in pixels: %u \n", image3DMaxDepth);

		cl_uint maxSamplers;
		clGetDeviceInfo(deviceId, CL_DEVICE_MAX_SAMPLERS, sizeof(cl_uint), &maxSamplers, &returnedSize);
		printf("Maximum number of samplers that can be used in a kernel. %u \n", maxSamplers);

	}
	cl_ulong localMemSize;
	clGetDeviceInfo(deviceId, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemSize, &returnedSize);
	printf("Size of local memory arena in bytes: %u \n", localMemSize);

	cl_uint maxClockFreq;
	clGetDeviceInfo(deviceId, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &maxClockFreq, &returnedSize);
	printf("Maximum configured clock frequency of the device: %u \n", maxClockFreq);

	cl_uint maxComputeUnits;
	clGetDeviceInfo(deviceId, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxComputeUnits, &returnedSize);
	printf("The number of parallel compute cores on the OpenCL device %u \n", maxComputeUnits);

	cl_uint maxConstantArgs;
	clGetDeviceInfo(deviceId, CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(cl_uint), &maxConstantArgs, &returnedSize);
	printf("Max number of arguments declared with the __constant qualifier in a kernel %u \n", maxConstantArgs);

	cl_ulong maxConstantBufferSize;
	clGetDeviceInfo(deviceId, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &maxConstantBufferSize, &returnedSize);
	printf("Max size in bytes of a constant buffer allocation. %X \n", maxConstantBufferSize);

	cl_ulong maxMemAllocationSize;
	clGetDeviceInfo(deviceId, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &maxMemAllocationSize, &returnedSize);
	printf("Max size of memory object allocation in bytes. %X \n", maxMemAllocationSize);

	size_t maxParamSize;
	clGetDeviceInfo(deviceId, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(size_t), &maxParamSize, &returnedSize);
	printf("Max size in bytes of the arguments that can be passed to a kernel. %i \n", maxParamSize);


	size_t maxWorkgroupSize;
	clGetDeviceInfo(deviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkgroupSize, &returnedSize);
	printf("Maximum number of work-items in a work-group executing a kernel using the data parallel execution model. %u \n", maxWorkgroupSize);

	size_t maxWorkItemSize[3];
	maxWorkItemSize[0] = 0;
	maxWorkItemSize[1] = 0;
	maxWorkItemSize[2] = 0;
	clGetDeviceInfo(deviceId, CL_DEVICE_MAX_WORK_ITEM_SIZES, 3*4, &maxWorkItemSize, &returnedSize);
	printf("Maximum number of work-items that can be specified in each dimension of the work-group. %u, %u, %u \n", maxWorkItemSize[0], maxWorkItemSize[1], maxWorkItemSize[2]);

	cl_uint maxWorkItemDimensions;
	clGetDeviceInfo(deviceId, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(size_t), &maxWorkItemDimensions, &returnedSize);
	printf("Maximum dimensions that specify the global and local work-item IDs used by the data parallel execution model. %u \n", maxWorkItemDimensions);
}