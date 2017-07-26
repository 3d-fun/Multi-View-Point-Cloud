#pragma once

#include "CL\cl.h"
#include <stdio.h>

/*
class CLContext
Structure that contains handles for the openCL paltform, device, context and CommandQueue.
Constructor will populate the fields automatically or throw an error.
*/
class ClContext
{
public:
	ClContext();
	virtual ~ClContext();

	cl_device_id deviceId;
	cl_platform_id platformId;
	cl_uint retNumDevices;
	cl_uint retNumPlatforms;
	cl_context context;
	cl_command_queue commandQueue;

private:
	void createContext();
	void createQueue();

	void printPlatformInfo(cl_platform_id platformId, cl_uint retNumPlatforms);
	void printDeviceInfo(cl_device_id deviceId);
};