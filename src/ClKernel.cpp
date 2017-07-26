#include "ClKernel.h"

ClKernel::ClKernel(const char* kernelSource, ClContext* context, std::vector<std::string> &kernelDefines)
{
	this->__context = context;
	FILE *fp;
	char *source_str;
	size_t source_size;

	fopen_s(&fp, kernelSource, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}

	fseek(fp, 0, SEEK_END);
	long fSize = ftell(fp);
	rewind(fp);

	long kernelDefinesSize = 0;
	for (int i = 0; i < kernelDefines.size(); i++)
	{
		kernelDefinesSize += kernelDefines[i].size();
	}
	size_t bufferSize = kernelDefinesSize + fSize;

	source_str = (char*)malloc(bufferSize);
	source_size = fread(source_str + kernelDefinesSize, 1, fSize, fp);
	fclose(fp);
	cl_int ret;

	int stringIndex = 0;
	for (int i = 0; i < kernelDefines.size(); i++)
	{
		for (int j = 0; j < kernelDefines[i].size(); j++)
		{
			source_str[stringIndex] = (kernelDefines[i])[j];
			stringIndex++;
		}
	}
	bufferSize = source_size + kernelDefinesSize;
	__program = NULL;
	__program = clCreateProgramWithSource(context->context, 1, (const char**)&source_str, (const size_t*)&bufferSize, &ret);
	switch (ret)
	{
		case CL_INVALID_CONTEXT:
		{
			printf("ERROR, context is invalid.\n");
			break;
		}
		case CL_INVALID_VALUE:
		{
			printf("ERROR, invalid value.\n");
			break;
		}
		case CL_OUT_OF_HOST_MEMORY:
		{
			printf("ERROR, out of host memory. \n");
			break;
		}
		case CL_SUCCESS:
		{
			printf("Success, program created. \n");
		}
	}

	ret = clBuildProgram(__program, 1, &context->deviceId, NULL, NULL, NULL);
	switch (ret)
	{
		case CL_INVALID_PROGRAM:
		{
			printf("ERROR, program is invalid. \n");
			break;
		}
		case CL_INVALID_VALUE:
		{
			printf("ERROR, invalid value.\n");
			break;
		}
		case CL_INVALID_DEVICE:
		{
			printf("ERROR, invalid device.\n");
			break;
		}
		case CL_INVALID_BINARY:
		{
			printf("ERROR, invalid binary program.\n");
			break;
		}
		case CL_INVALID_BUILD_OPTIONS:
		{
			printf("ERROR, invalid build options.\n");
			break;
		}
		case CL_INVALID_OPERATION:
		{
			printf("ERROR, invalid build operation.\n");
			break;
		}
		case CL_COMPILER_NOT_AVAILABLE:
		{
			printf("ERROR, compiler not available.\n");
			break;
		}
		case CL_OUT_OF_HOST_MEMORY:
		{
			printf("ERROR, out of host memory.\n");
			break;
		}
		case CL_BUILD_PROGRAM_FAILURE:
		{
			printf("ERROR, build failed.\n");
			break;
		}
		case CL_SUCCESS:
		{
			printf("Success, build successful.\n");
			break;
		}

	}

	size_t buildLogSize;
	clGetProgramBuildInfo(__program, context->deviceId, CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize);
	char* buildLog = new char[buildLogSize];
	clGetProgramBuildInfo(__program, context->deviceId, CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog, &buildLogSize);

	printf(buildLog);
	printf("\n");
	delete buildLog;

	free(source_str);

	__globalWorkSize = 0;
	__localWorkSize = 0;
}
ClKernel::~ClKernel()
{
	for (int i = 0; i < __kernel.size(); i++)
	{
		clReleaseKernel(__kernel[i]);
	}
	__kernel.clear();

	clReleaseProgram(__program);
	for (int i = 0; i < __memoryObjects.size(); i++)
	{
		clReleaseMemObject(__memoryObjects[i]);
	}
	__memoryObjects.clear();
}

void ClKernel::train()
{

}

int ClKernel::createKernel(const char* kernelName)
{
	cl_int ret;
	cl_kernel k = clCreateKernel(__program, kernelName, &ret);
	int kernelPos = -1;

	switch (ret)
	{
	case CL_INVALID_PROGRAM:
		printf("ERROR, program is not a valid program object. \n");
		break;

	case CL_INVALID_PROGRAM_EXECUTABLE:
		printf("ERROR, there is no successfully built executable for program. \n");
		break;

	case CL_INVALID_KERNEL_NAME:
		printf("ERROR, 'kernelName' is not found in the program. \n");
		break;

	case CL_INVALID_KERNEL_DEFINITION:
		printf("ERROR, invalid kernel definition. \n");
		break;

	case CL_INVALID_VALUE:
		printf("ERROR, 'kernelName' is NULL. \n");
		break;

	case CL_OUT_OF_HOST_MEMORY:
		printf("ERROR, out of host memory. \n");
		break;

	case CL_SUCCESS:
		printf("Success, kernel created. \n");
		__kernel.push_back(k);
		kernelPos = __kernel.size();
		break;
	}

	char kernelInfo[1000];
	size_t kernelInfoSize;

	clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, 1000, kernelInfo, &kernelInfoSize);
	OutputDebugStringA("Kernel function name: ");
	OutputDebugStringA(kernelInfo);
	OutputDebugStringA("\n");

	return kernelPos;
}

void ClKernel::addKernelArg(size_t kernelIndex, int argId, unsigned int bufferSize, void* buffer)
{
	if (__kernel.size() <= kernelIndex)
	{
		printf("ERROR, kernel out of index. \n");
		return;
	}
	cl_int ret = clSetKernelArg(__kernel[kernelIndex], argId, bufferSize, buffer);

	switch (ret)
	{
	case CL_INVALID_KERNEL:
		printf("ERROR, invalid kernel object. \n");
		break;

	case CL_INVALID_ARG_INDEX:
		printf("ERROR, 'argId' is not a valid argument index. \n");
		break;

	case CL_INVALID_ARG_VALUE:
		printf("ERROR, 'bufer' is NULL. \n");
		break;

	case CL_INVALID_MEM_OBJECT:
		printf("ERROR, memory buffer is not a valid memory object. \n");
		break;

	case CL_INVALID_SAMPLER:
		printf("ERROR, sampler buffer is not a valid sampler object. \n");
		break;

	case CL_INVALID_ARG_SIZE:
		printf("ERROR, invalid argument size. \n");
		break;

	case CL_SUCCESS:
		//printf("Success, kernel argument added. \n");
		break;
	
	}
}

void ClKernel::createBuffer(cl_context context, cl_mem_flags memFlag, size_t bufferSize, cl_mem &memObj)
{
	cl_int ret;
	memObj = clCreateBuffer(context, memFlag, bufferSize, NULL, &ret);

	switch (ret)
	{
	case CL_SUCCESS:

		break;

	case CL_INVALID_CONTEXT:
		printf("ERROR, context is not valid");
		break;

	case CL_INVALID_VALUE:
		printf("ERROR, memory flags are not valid");
		break;

	case CL_INVALID_BUFFER_SIZE:
		printf("ERROR, invalid buffer size");
		break;

	case CL_INVALID_HOST_PTR:
		printf("ERROR, host pointer is invalid");
		break;

	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
		printf("ERROR, memory allocation failure");
		break;

	case CL_OUT_OF_HOST_MEMORY:
		printf("ERROR, out of host memory");
		break;
	}
}