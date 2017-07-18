#pragma once

#include <fstream>
#include "ClContext.h"
#include <stdio.h>
#include <vector>
#include <windows.h>

/*
class ClKernel
Base Kernel class. Contains kernel handles and functions used to enqueue kernels and add arguments.
Loads the kernel source directory 'kernelSource' and builds the source. 

contructor :
@param const char* kernel source. 
@param ClContext* context.
*/
class ClKernel
{
public:
	ClKernel(const char* kernelSource, ClContext* context, std::vector<std::string> &kernelDefines);
	virtual ~ClKernel();
	virtual void train();
	size_t totalWorkItems;
protected:
	/*
	int createKernel
	creates a kernel object of 'kernelName'. 'kerneName' must be located in the source.

	@param const char* kernelName
	@return int, index of the kernel created.
	*/
	int createKernel(const char* kernelName);
	
	/*
	void addKernelArg 
	adds an argument to the kernel in the kernel vector, indexed by kernelIndex.

	@param size_t kernelIndex: Index of the kernel in the kernel vector.
	@param int argId : argument index
	@param unsigned int bufferSize
	@param void* buffer.
	*/
	void addKernelArg(size_t kernelIndex, int argId, unsigned int bufferSize, void* buffer);
	
	/*
	void enqueueKernel
	enqueues and runs the specified kernel with 'totalWorkItems' work items.

	@param size_t kernelIndex.
	*/
	inline void enqueueKernel(size_t kernelIndex)
	{
		//compute the number of iterations, offset, global work items and local work items

		if (__kernel.size() <= kernelIndex)
		{
			printf("ERROR, kernel out of index. \n");
			return;
		}

		size_t returnSize;
		size_t maxWorkItemSize[3];
		maxWorkItemSize[0] = 0;
		maxWorkItemSize[1] = 0;
		maxWorkItemSize[2] = 0;
		clGetDeviceInfo(__context->deviceId, CL_DEVICE_MAX_WORK_ITEM_SIZES, 3 * sizeof(cl_int), &maxWorkItemSize, &returnSize);

		__localWorkSize = 1;
		size_t workItemsPerIteration = maxWorkItemSize[0];
		size_t iterations = (totalWorkItems / workItemsPerIteration) + 1;
		size_t offset = 0;

		size_t* workItemIteration = new size_t[iterations];
		int i;
		for (i = 0; i < iterations - 1; i++)
		{
			workItemIteration[i] = workItemsPerIteration;
		}
		workItemIteration[i] = totalWorkItems % workItemsPerIteration;

		for (int i = 0; i < iterations; i++)
		{
			__globalWorkSize = workItemIteration[i];

			cl_int ret = clEnqueueNDRangeKernel(__context->commandQueue, __kernel[kernelIndex], 1, &offset, &__globalWorkSize, &__localWorkSize, 0, NULL, NULL);
			switch (ret)
			{
			case CL_INVALID_PROGRAM_EXECUTABLE:
				printf("ERROR, no successfully built program executable available for device. \n");
				break;

			case CL_INVALID_COMMAND_QUEUE:
				printf("ERROR, invalid command queue. \n");
				break;

			case CL_INVALID_KERNEL:
				printf("ERROR, invalid kernel. \n");
				break;

			case CL_INVALID_CONTEXT:
				printf("ERROR, invalid context. \n");
				break;

			case CL_INVALID_KERNEL_ARGS:
				printf("ERROR, the kernel arguments have not been specified. \n");
				break;

			case CL_INVALID_WORK_DIMENSION:
				printf("ERROR, invalid work dimensions. \n");
				break;

			case CL_INVALID_GLOBAL_WORK_SIZE:
				printf("ERROR, invalid work size. \n");
				break;

			case CL_INVALID_GLOBAL_OFFSET:
				printf("ERROR, invalid global offset. \n");
				break;

			case CL_INVALID_WORK_GROUP_SIZE:
				printf("ERROR, invalid work group size. \n");
				break;

			case CL_INVALID_WORK_ITEM_SIZE:
				printf("ERROR, invalid work item size. \n");
				break;

			case CL_MISALIGNED_SUB_BUFFER_OFFSET:
				printf("ERROR, misaligned sub buffer offset. \n");
				break;


			case CL_INVALID_IMAGE_SIZE:
				printf("ERROR, invalid image size. \n");
				break;

			case CL_OUT_OF_RESOURCES:
				printf("ERROR, out of resources. \n");
				break;

			case CL_MEM_OBJECT_ALLOCATION_FAILURE:
				printf("ERROR, memory object allocation failure. \n");
				break;

			case CL_INVALID_EVENT_WAIT_LIST:
				printf("ERROR, invalid wait list. \n");
				break;

			case CL_OUT_OF_HOST_MEMORY:
				printf("ERROR, out of host memory. \n");
				break;

			case CL_SUCCESS:
				//printf("Success, kernel enqueued. \n");
				break;
			}

			offset = offset + __globalWorkSize;

		}
		clFinish(__context->commandQueue);
		delete workItemIteration;
	}

	inline void enqueueKernel(size_t kernelIndex, size_t workItemSizeX, size_t workItemSizeY)
	{
		//compute the number of iterations, offset, global work items and local work items

		if (__kernel.size() <= kernelIndex)
		{
			printf("ERROR, kernel out of index. \n");
			return;
		}

		size_t returnSize;
		size_t maxWorkItemSize[3];
		maxWorkItemSize[0] = 0;
		maxWorkItemSize[1] = 0;
		maxWorkItemSize[2] = 0;
		clGetDeviceInfo(__context->deviceId, CL_DEVICE_MAX_WORK_ITEM_SIZES, 3 * sizeof(cl_int), &maxWorkItemSize, &returnSize);

		totalWorkItems = workItemSizeX*workItemSizeY;

		__localWorkSize = 1;
		size_t workItemsPerIteration = maxWorkItemSize[0];

		size_t iterations = (totalWorkItems / workItemsPerIteration) + 1;
		size_t offset = 0;

		size_t workItemIteration[1024];
		int i;
		for (i = 0; i < iterations - 1; i++)
		{
			workItemIteration[i] = workItemsPerIteration;
		}
		workItemIteration[i] = totalWorkItems % workItemsPerIteration;

		for (int i = 0; i < iterations; i++)
		{
			__globalWorkSize = workItemIteration[i];

			cl_int ret = clEnqueueNDRangeKernel(__context->commandQueue, __kernel[kernelIndex], 1, &offset, &__globalWorkSize, &__localWorkSize, 0, NULL, NULL);
			//cl_int ret = clEnqueueNDRangeKernel(__context->commandQueue, __kernel[kernelIndex], 1, &offset, &__globalWorkSize, &__localWorkSize, i, k_events, &k_events[i]);
			switch (ret)
			{
			case CL_INVALID_PROGRAM_EXECUTABLE:
				printf("ERROR, no successfully built program executable available for device. \n");
				break;

			case CL_INVALID_COMMAND_QUEUE:
				printf("ERROR, invalid command queue. \n");
				break;

			case CL_INVALID_KERNEL:
				printf("ERROR, invalid kernel. \n");
				break;

			case CL_INVALID_CONTEXT:
				printf("ERROR, invalid context. \n");
				break;

			case CL_INVALID_KERNEL_ARGS:
				printf("ERROR, the kernel arguments have not been specified. \n");
				break;

			case CL_INVALID_WORK_DIMENSION:
				printf("ERROR, invalid work dimensions. \n");
				break;

			case CL_INVALID_GLOBAL_WORK_SIZE:
				printf("ERROR, invalid work size. \n");
				break;

			case CL_INVALID_GLOBAL_OFFSET:
				printf("ERROR, invalid global offset. \n");
				break;

			case CL_INVALID_WORK_GROUP_SIZE:
				printf("ERROR, invalid work group size. \n");
				break;

			case CL_INVALID_WORK_ITEM_SIZE:
				printf("ERROR, invalid work item size. \n");
				break;

			case CL_MISALIGNED_SUB_BUFFER_OFFSET:
				printf("ERROR, misaligned sub buffer offset. \n");
				break;


			case CL_INVALID_IMAGE_SIZE:
				printf("ERROR, invalid image size. \n");
				break;

			case CL_OUT_OF_RESOURCES:
				printf("ERROR, out of resources. \n");
				break;

			case CL_MEM_OBJECT_ALLOCATION_FAILURE:
				printf("ERROR, memory object allocation failure. \n");
				break;

			case CL_INVALID_EVENT_WAIT_LIST:
				printf("ERROR, invalid wait list. \n");
				break;

			case CL_OUT_OF_HOST_MEMORY:
				printf("ERROR, out of host memory. \n");
				break;

			case CL_SUCCESS:
				//printf("Success, kernel enqueued. \n");
				break;
			}

			offset = offset + __globalWorkSize;
			clFinish(__context->commandQueue);
		}
		
	}

	void createBuffer(cl_context context, cl_mem_flags memFlag, size_t bufferSize, cl_mem &memObj);

	cl_program __program;
	std::vector<cl_kernel> __kernel;
	ClContext* __context;
	std::vector<cl_mem> __memoryObjects;
private:
	size_t __globalWorkSize;
	size_t __localWorkSize;

};