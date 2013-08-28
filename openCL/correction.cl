/*
 * Performs correction of raw image with dark/flats.
 *
 * @param raw: Pointer to global memory with the raw image
 * @param raw: Pointer to global memory with the dark_data image
 * @param raw: Pointer to global memory with the dark_ref image
 * @param raw: Pointer to global memory with the flat1 image
 * @param raw: Pointer to global memory with the flat2 image
 * @param raw: Pointer to global memory with the output image
 * @param width: images width
 * @param height: images height
 *
 *
*/

__kernel void correction(
	__global float* raw,
	__global float* dark_data,
	__global float* dark_ref,
	__global float* flat1,
	__global float* flat2,
	__global float* output,
	int width,
	int height)
{

	int gid0 = get_global_id(0);
	int gid1 = get_global_id(1);

	if (gid1 < width && gid0 < height) {

		int pos = gid1*width+gid0;
		float d = flat1[pos]+flat2[pos] - 2.0f*dark_data[pos];
		if (d == 0) d = 1.0f;
		output[pos] = (raw[pos]-dark_ref[pos])/d;
	}
}
