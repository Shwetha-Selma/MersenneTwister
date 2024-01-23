/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This sample demonstrates the use of CURAND to generate
 * random numbers on GPU 
 */

// Utilities and system includes
// includes, system
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dpct/rng_utils.hpp>

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cmath>

const int DEFAULT_RAND_N = 2400000;
const unsigned int DEFAULT_SEED = 777;

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  // Start logs
  printf("%s Starting...\n\n", argv[0]);

  // initialize the GPU, either identified by --device
  // or by picking the device with highest flop rate.

  // parsing the number of random numbers to generate
  int rand_n = DEFAULT_RAND_N;

  if (checkCmdLineFlag(argc, (const char **)argv, "count")) {
    rand_n = getCmdLineArgumentInt(argc, (const char **)argv, "count");
  }

  printf("Allocating data for %i samples...\n", rand_n);

  // parsing the seed
  int seed = DEFAULT_SEED;

  if (checkCmdLineFlag(argc, (const char **)argv, "seed")) {
    seed = getCmdLineArgumentInt(argc, (const char **)argv, "seed");
  }

  printf("Seeding with %i ...\n", seed);

  dpct::queue_ptr stream;
  /*
  DPCT1025:14: The SYCL queue is created ignoring the flag and priority options.
  */
  checkCudaErrors(
      DPCT_CHECK_ERROR(stream = dpct::get_current_device().create_queue()));

  float *d_Rand;
  checkCudaErrors(DPCT_CHECK_ERROR(
      d_Rand = sycl::malloc_device<float>(rand_n, dpct::get_in_order_queue())));

  dpct::rng::host_rng_ptr prngGPU;

  checkCudaErrors(DPCT_CHECK_ERROR(prngGPU = dpct::rng::create_host_rng(
                                       dpct::rng::random_engine_type::mt2203)));
  
  checkCudaErrors(DPCT_CHECK_ERROR(prngGPU->set_queue(stream)));
  checkCudaErrors(DPCT_CHECK_ERROR(prngGPU->set_seed(seed)));

  //
  // Example 1: Compare random numbers generated on GPU 
  float *h_RandGPU;
  checkCudaErrors(DPCT_CHECK_ERROR(h_RandGPU = sycl::malloc_host<float>(
                                       rand_n, dpct::get_in_order_queue())));

  printf("Generating random numbers on GPU...\n\n");
  checkCudaErrors(
      DPCT_CHECK_ERROR(prngGPU->generate_uniform((float *)d_Rand, rand_n)));

  printf("\nReading back the results...\n");
  checkCudaErrors(DPCT_CHECK_ERROR(
      stream->memcpy(h_RandGPU, d_Rand, rand_n * sizeof(float))));

  printf("Shutting down...\n");

  checkCudaErrors(DPCT_CHECK_ERROR(prngGPU.reset()));
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_current_device().destroy_queue(stream)));
  checkCudaErrors(
      DPCT_CHECK_ERROR(sycl::free(d_Rand, dpct::get_in_order_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(sycl::free(h_RandGPU, dpct::get_in_order_queue())));

  
}
