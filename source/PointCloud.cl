#define SAMPLE_WINDOW_SIZE 9
#define LOCAL_CAMERA_SIZE 2
#define BRIGHTNESS_THRESHOLD 24

typedef struct _Matrix
{
	float m[16];
}Matrix;

typedef struct _Vector
{
	float v[4];
}Vector;

inline void multMM(Matrix* m1, Matrix* m2, Matrix* result);
inline void multMV(Matrix* m, Vector* v, Vector* result);
inline void getImageIndices(int globalIndex, int* x, int* y, int* imageIndex, int imageWidth, int imageHeight);
inline void sampleImage(__read_only image3d_t stereoImages, int x, int y, int imageIndex, int sampleWindow, uchar* sample);
inline void cameraSubset(int centerCamera, int subsetSize, int numCameras, int* subset);
inline void getLocalDepthRange(Matrix* cameraMatrix, __global float* globalRange, float* depthRange);
inline bool isClipped(Vector* screenCoord, int imageWidth, int imageHeight);
inline float normalizedCrossCorrelation(unsigned char* v0, unsigned char* v1, int numSamples);
inline float crossCorrelation(float* v0, float* v1, int numSamples);
inline void subtractMean(unsigned char* v0, float* result, int numSamples);

inline void multMM(Matrix* m1, Matrix* m2, Matrix* result)
{
	result->m[0] = m1->m[0]*m2->m[0] + m1->m[1]*m2->m[4] + m1->m[2]*m2->m[8] + m1->m[3]*m2->m[12];
	result->m[1] = m1->m[0]*m2->m[1] + m1->m[1]*m2->m[5] + m1->m[2]*m2->m[9] + m1->m[3]*m2->m[13];
	result->m[2] = m1->m[0]*m2->m[2] + m1->m[1]*m2->m[6] + m1->m[2]*m2->m[10] + m1->m[3]*m2->m[14];
	result->m[3] = m1->m[0]*m2->m[3] + m1->m[1]*m2->m[7] + m1->m[2]*m2->m[11] + m1->m[3]*m2->m[15];

	result->m[4] = m1->m[4]*m2->m[0] + m1->m[5]*m2->m[4] + m1->m[6]*m2->m[8] + m1->m[7]*m2->m[12];
	result->m[5] = m1->m[4]*m2->m[1] + m1->m[5]*m2->m[5] + m1->m[6]*m2->m[9] + m1->m[7]*m2->m[13];
	result->m[6] = m1->m[4]*m2->m[2] + m1->m[5]*m2->m[6] + m1->m[6]*m2->m[10] + m1->m[7]*m2->m[14];
	result->m[7] = m1->m[4]*m2->m[3] + m1->m[5]*m2->m[7] + m1->m[6]*m2->m[11] + m1->m[7]*m2->m[15];

	result->m[8] = m1->m[8]*m2->m[0] + m1->m[9]*m2->m[4] + m1->m[10]*m2->m[8] + m1->m[11]*m2->m[12];
	result->m[9] = m1->m[8]*m2->m[1] + m1->m[9]*m2->m[5] + m1->m[10]*m2->m[9] + m1->m[11]*m2->m[13];
	result->m[10] = m1->m[8]*m2->m[2] + m1->m[9]*m2->m[6] + m1->m[10]*m2->m[10] + m1->m[11]*m2->m[14];
	result->m[11] = m1->m[8]*m2->m[3] + m1->m[9]*m2->m[7] + m1->m[10]*m2->m[11] + m1->m[11]*m2->m[15];

	result->m[12] = m1->m[12]*m2->m[0] + m1->m[13]*m2->m[4] + m1->m[14]*m2->m[8] + m1->m[15]*m2->m[12];
	result->m[13] = m1->m[12]*m2->m[1] + m1->m[13]*m2->m[5] + m1->m[14]*m2->m[9] + m1->m[15]*m2->m[13];
	result->m[14] = m1->m[12]*m2->m[2] + m1->m[13]*m2->m[6] + m1->m[14]*m2->m[10] + m1->m[15]*m2->m[14];
	result->m[15] = m1->m[12]*m2->m[3] + m1->m[13]*m2->m[7] + m1->m[14]*m2->m[11] + m1->m[15]*m2->m[15];
}
inline void multMV(Matrix* m, Vector* v, Vector* result)
{
	result->v[0] = m->m[0]*v->v[0] + m->m[1]*v->v[1] + m->m[2]*v->v[2] + m->m[3]*v->v[3];
	result->v[1] = m->m[4]*v->v[0] + m->m[5]*v->v[1] + m->m[6]*v->v[2] + m->m[7]*v->v[3];
	result->v[2] = m->m[8]*v->v[0] + m->m[9]*v->v[1] + m->m[10]*v->v[2] + m->m[11]*v->v[3];
	result->v[3] = m->m[12]*v->v[0] + m->m[13]*v->v[1] + m->m[14]*v->v[2] + m->m[15]*v->v[3];
}
inline void getImageIndices(int globalIndex, int* x, int* y, int* imageIndex, int imageWidth, int imageHeight)
{
	int index = globalIndex;
	*imageIndex = (index / (imageWidth*imageHeight));
	index = index - (*imageIndex)*imageWidth*imageHeight;
	*y = (index / imageWidth);
	index = index - (*y)*imageWidth;
	*x = index;
}

inline void sampleImage(__read_only image3d_t stereoImages, int x, int y, int imageIndex, int sampleWindow, uchar* sample)
{
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP;
	int m = sampleWindow / 2;
	int startX = x - m;
	int startY = y - m;
	int endX = startX + sampleWindow;
	int endY = startY + sampleWindow;
	int4 coord;

	int sampleIndex = 0;
	for (int sampleX = startX; sampleX < endX; sampleX++)
	{
		for (int sampleY = startY; sampleY < endY; sampleY++)
		{
			coord = (int4)(sampleX,sampleY,imageIndex,0);
			uint4 s = read_imageui(stereoImages, sampler, coord);	
			sample[sampleIndex] = (uchar)s.x;
			sample[sampleIndex + 1] = (uchar)s.y;
			sample[sampleIndex + 2] = (uchar)s.z;
			sampleIndex = sampleIndex + 3;
		}
	}
}

inline void cameraSubset(int centerCamera, int subsetSize, int numCameras, int* subset)
{
	int n = subsetSize / 2;
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

inline void getLocalDepthRange(Matrix* cameraMatrix, __global float* globalRange, float* depthRange)
{
	float maxD;
	float minD;

	Vector globalPosition;
	Vector screenPos;

	globalPosition.v[0] = globalRange[0];
	globalPosition.v[1] = globalRange[1];
	globalPosition.v[2] = globalRange[2];
	globalPosition.v[3] = 1.0;
	multMV(cameraMatrix, &globalPosition, &screenPos);
	maxD = screenPos.v[2];
	minD = screenPos.v[2];


	globalPosition.v[0] = globalRange[0];
	globalPosition.v[1] = globalRange[1];
	globalPosition.v[2] = globalRange[5];
	globalPosition.v[3] = 1.0;
	multMV(cameraMatrix, &globalPosition, &screenPos);
	maxD = max(screenPos.v[2], maxD);
	minD = min(screenPos.v[2], minD);


	globalPosition.v[0] = globalRange[0];
	globalPosition.v[1] = globalRange[4];
	globalPosition.v[2] = globalRange[2];
	globalPosition.v[3] = 1.0;
	multMV(cameraMatrix, &globalPosition, &screenPos);
	maxD = max(screenPos.v[2], maxD);
	minD = min(screenPos.v[2], minD);


	globalPosition.v[0] = globalRange[0];
	globalPosition.v[1] = globalRange[4];
	globalPosition.v[2] = globalRange[5];
	globalPosition.v[3] = 1.0;
	multMV(cameraMatrix, &globalPosition, &screenPos);
	maxD = max(screenPos.v[2], maxD);
	minD = min(screenPos.v[2], minD);


	globalPosition.v[0] = globalRange[3];
	globalPosition.v[1] = globalRange[1];
	globalPosition.v[2] = globalRange[2];
	globalPosition.v[3] = 1.0;
	multMV(cameraMatrix, &globalPosition, &screenPos);
	maxD = max(screenPos.v[2], maxD);
	minD = min(screenPos.v[2], minD);


	globalPosition.v[0] = globalRange[3];
	globalPosition.v[1] = globalRange[1];
	globalPosition.v[2] = globalRange[5];
	globalPosition.v[3] = 1.0;
	multMV(cameraMatrix, &globalPosition, &screenPos);
	maxD = max(screenPos.v[2], maxD);
	minD = min(screenPos.v[2], minD);


	globalPosition.v[0] = globalRange[3];
	globalPosition.v[1] = globalRange[4];
	globalPosition.v[2] = globalRange[2];
	globalPosition.v[3] = 1.0;
	multMV(cameraMatrix, &globalPosition, &screenPos);
	maxD = max(screenPos.v[2], maxD);
	minD = min(screenPos.v[2], minD);


	globalPosition.v[0] = globalRange[3];
	globalPosition.v[1] = globalRange[4];
	globalPosition.v[2] = globalRange[5];
	globalPosition.v[3] = 1.0;
	multMV(cameraMatrix, &globalPosition, &screenPos);
	maxD = max(screenPos.v[2], maxD);
	minD = min(screenPos.v[2], minD);

	depthRange[0] = minD;
	depthRange[1] = maxD;
}

inline bool isClipped(Vector* screenCoord, int imageWidth, int imageHeight)
{
	if (screenCoord->v[0] >= (float)imageWidth || screenCoord->v[0] < 0.0)
		return true;
	if (screenCoord->v[1] >= (float)imageHeight || screenCoord->v[1] < 0.0)
		return true;
	return false;
}

inline float normalizedCrossCorrelation(unsigned char* v0, unsigned char* v1, int numSamples)
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

inline float crossCorrelation(float* v0, float* v1, int numSamples)
{
	float num = 0.0;
	float den1 = 0.0;
	float den2 = 0.0;

	for(int i=0; i < numSamples*3; i++)
	{
		num += v0[i]*v1[i];
		den1 += v0[i]*v0[i];
		den2 += v1[i]*v1[i];
	}
	float den = den1*den2;
	if(den == 0.0)
		return 0.0;
	return num / (sqrt(den));
}

inline void subtractMean(unsigned char* v0, float* result, int numSamples)
{
	float meanV0[] = { 0.0, 0.0, 0.0 };

	for (int i = 0; i < numSamples; i++)
	{
		meanV0[0] += (float)v0[i * 3];
		meanV0[1] += (float)v0[i * 3 + 1];
		meanV0[2] += (float)v0[i * 3 + 2];
	}
	meanV0[0] = meanV0[0] / (float)numSamples;
	meanV0[1] = meanV0[1] / (float)numSamples;
	meanV0[2] = meanV0[2] / (float)numSamples;

	for (int i = 0; i < numSamples; i++)
	{
		result[i * 3] = (float)v0[i*3] - meanV0[0];
		result[i * 3 + 1] = (float)v0[i*3 + 1] - meanV0[1];
		result[i * 3 + 2] = (float)v0[i*3 + 2] - meanV0[2];
	}
}

__kernel void computeDepth(__read_only image3d_t stereoImages, __global Matrix* transformMatrix, __global Matrix* tranformMatrixInv, __global float* depthSearchRange, 
	__global int* numValidDepthSamples, __global float* depthSample, __global float* depthCorrelation, int imageReferenceIndex, int rayMarchIndex)
{
	int x, y, imageIndex;
	int id = get_global_id(0);
	int imageWidth, imageHeight, numImages;
	int4 imageDim = get_image_dim(stereoImages);
	imageWidth = imageDim.x;
	imageHeight = imageDim.y;
	numImages = imageDim.z;

	uchar windowSample_i[SAMPLE_WINDOW_SIZE*SAMPLE_WINDOW_SIZE*3];
	uchar windowSample_j[SAMPLE_WINDOW_SIZE*SAMPLE_WINDOW_SIZE*3];

	float windowSampleFloat_i[SAMPLE_WINDOW_SIZE*SAMPLE_WINDOW_SIZE*3];
	float windowSampleFloat_j[SAMPLE_WINDOW_SIZE*SAMPLE_WINDOW_SIZE*3];

	getImageIndices(id, &x, &y, &imageIndex, imageWidth, imageHeight);
	imageIndex = imageReferenceIndex;

	Vector screenPos;
	Vector globalPos;
	Matrix transform = transformMatrix[imageIndex];
	Matrix TransformInv = tranformMatrixInv[imageIndex];

	float depthRange[] = {depthSearchRange[imageIndex*2], depthSearchRange[imageIndex*2 + 1]};
	float depthStep = (depthRange[1] - depthRange[0])/((float)RAY_MARCH_ITERS);

	sampleImage(stereoImages, x, y, imageIndex, SAMPLE_WINDOW_SIZE, windowSample_i);

	int sampleSum = 0;
	int imageCompareSet[LOCAL_CAMERA_SIZE];
	for(int i=0; i < SAMPLE_WINDOW_SIZE*SAMPLE_WINDOW_SIZE*3; i++)
	{
		sampleSum += windowSample_i[i];
	}

	int acceptedNcc = 0;
	float nccSum = 0.0;

	if(sampleSum >= 3*BRIGHTNESS_THRESHOLD*SAMPLE_WINDOW_SIZE*SAMPLE_WINDOW_SIZE)
	{
		float d = depthRange[0] + depthStep*rayMarchIndex;

		Vector screenCoordReference;
		screenCoordReference.v[0] = (float)x*d;
		screenCoordReference.v[1] = (float)y*d;
		screenCoordReference.v[2] = d;
		screenCoordReference.v[3] = 1.0;

		Vector screenCoord_j;

		int acceptedNcc = 0;

		subtractMean(windowSample_i, windowSampleFloat_i, SAMPLE_WINDOW_SIZE*SAMPLE_WINDOW_SIZE);
		cameraSubset(imageIndex, LOCAL_CAMERA_SIZE, numImages, imageCompareSet);
		multMV(&TransformInv, &screenCoordReference, &globalPos);
		
		for (int localCameraIndex = 0; localCameraIndex < LOCAL_CAMERA_SIZE; localCameraIndex++)
		{
			int imageIndex_j = imageCompareSet[localCameraIndex];
			if(imageIndex_j  != imageIndex)
			{
				Matrix transform_j = transformMatrix[imageIndex_j];
				multMV(&transform_j, &globalPos, &screenCoord_j);
				screenCoord_j.v[0] = screenCoord_j.v[0] / screenCoord_j.v[2];
				screenCoord_j.v[1] = screenCoord_j.v[1] / screenCoord_j.v[2];
				screenCoord_j.v[2] = screenCoord_j.v[2] / screenCoord_j.v[2];
				
				if(!isClipped(&screenCoord_j, imageWidth, imageHeight))
				{
					int sampleX_j = (int)round(screenCoord_j.v[0]);
					int sampleY_j = (int)round(screenCoord_j.v[1]);

					sampleImage(stereoImages, sampleX_j, sampleY_j, imageIndex_j, SAMPLE_WINDOW_SIZE, windowSample_j);
					subtractMean(windowSample_j, windowSampleFloat_j, SAMPLE_WINDOW_SIZE*SAMPLE_WINDOW_SIZE);

					//float ncc = normalizedCrossCorrelation(windowSample_i, windowSample_j, SAMPLE_WINDOW_SIZE*SAMPLE_WINDOW_SIZE);
					float ncc = crossCorrelation(windowSampleFloat_i, windowSampleFloat_j, SAMPLE_WINDOW_SIZE*SAMPLE_WINDOW_SIZE);

					if (ncc >= NCC_THRESHOLD)
					{
						nccSum += ncc;
						acceptedNcc++;
					}
				}	
			}
		}

		if (acceptedNcc >= 2)
		{
			nccSum = nccSum / (float)acceptedNcc;
			float corr = depthCorrelation[id];
			if(nccSum > corr)
			{
				depthSample[id] = d;
				numValidDepthSamples[id] = numValidDepthSamples[id] + 1;
				depthCorrelation[id] = nccSum;
			}
		}
	}
}