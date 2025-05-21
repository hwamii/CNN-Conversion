/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/
#include "host.h"
#include "experimental/xrt_profile.h"
#include "xcl2.hpp"
#include <vector>
#include "helper/read_write_csv.hpp"
#include "cnn_sw/top.hpp"

using namespace std;
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    // Create a user_range using the shortcut constructor.  This will
    //  start measuring the time immediately
    xrt::profile::user_range range("Phase 1", "Start of execution to context creation");

    xrt::profile::user_event events;

    std::string binaryFile = argv[1];

    // Allocate result buffer on host memory
    size_t vector_size_bytes = sizeof(int) * LENGTH;
    std::vector<float, aligned_allocator<float> > input(16); //make multiple sources //source weight will be(16*512)
    std::vector<float, aligned_allocator<float> > iWeights_0(16 * 512);
    std::vector<float, aligned_allocator<float> > iBias_0(512);
    //Layer1
    std::vector<float, aligned_allocator<float> > iWeights_1(512 * 256);
    std::vector<float, aligned_allocator<float> > iBias_1(256);
    //Layer2
    std::vector<float, aligned_allocator<float> > iWeights_2(256 * 64);
    std::vector<float, aligned_allocator<float> > iBias_2(64);
    //Layer3
    std::vector<float, aligned_allocator<float> > iWeights_3(64 * 2);
    std::vector<float, aligned_allocator<float> > iBias_3(2);
    //Result
    std::vector<float, aligned_allocator<float> > oData(2);
    std::vector<int, aligned_allocator<int> > result_sim(LENGTH);
    std::vector<float, aligned_allocator<float> > result_krnl(LENGTH);


    // Read essential input and parameter data
    string base_dir = "/home/root/ML_spectrum";
   
    readFP<float>(base_dir +  "/SNR0N16/occupied4TestingSNR5.csv",&input[0],0,1,16);
    readFP<float>(base_dir +  "/dense/dense_weights.csv",&iWeights_0[0],0,16,512);
    readFP<float>(base_dir +  "/dense/dense_biases.csv",&iBias_0[0],0,1,512);
    readFP<float>(base_dir +  "/dense1/dense_1_weights.csv",&iWeights_1[0],0,512,256);
    readFP<float>(base_dir +  "/dense1/dense_1_biases.csv",&iBias_1[0],0,1,256);
    readFP<float>(base_dir +  "/dense2/dense_2_weights.csv",&iWeights_2[0],0,256,64);
    readFP<float>(base_dir +  "/dense2/dense_2_biases.csv",&iBias_2[0],0,1,64);
    readFP<float>(base_dir +  "/dense3/dense_3_weights.csv",&iWeights_3[0],0,64,2);
    readFP<float>(base_dir +  "/dense3/dense_3_biases.csv",&iBias_3[0],0,1,2);



    range.end();

    events.mark("Test data created");

    range.start("Phase 2", "Context creation and loading of xclbin");

    // OPENCL HOST CODE AREA START
    cl_int err;
    cl::Context context;
    cl::CommandQueue q;
    cl::Kernel krnl;
    auto devices = xcl::get_xil_devices();
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl = cl::Kernel(program, "spectrumSensing_top", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    range.end();
    events.mark("Context created and Xclbin loaded");
    range.start("Phase 3", "Kernel and buffer creation");

    //
    OCL_CHECK(err, cl::Buffer input_buff(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 16 * sizeof(float),
                                       input.data(), &err)); //input

    //Layer0
    OCL_CHECK(err, cl::Buffer iWbuffer_0(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 16 * 512 * sizeof(float),
                                           iWeights_0.data(), &err)); //change this to reflect project and same #
    OCL_CHECK(err, cl::Buffer iBbuffer_0(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 512 * sizeof(float),
    										iBias_0.data(), &err)); //change this to reflect project and same #
    //Layer1
    OCL_CHECK(err, cl::Buffer iWbuffer_1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 512 * 256 * sizeof(float),
                                               iWeights_1.data(), &err)); //change this to reflect project and same #
    OCL_CHECK(err, cl::Buffer iBbuffer_1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 256 * sizeof(float),
                                               iBias_1.data(), &err)); //change this to reflect project and same #
    //Layer2
    OCL_CHECK(err, cl::Buffer iWbuffer_2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 256 * 64 * sizeof(float),
                                               iWeights_2.data(), &err)); //change this to reflect project and same #
    OCL_CHECK(err, cl::Buffer iBbuffer_2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 64 * sizeof(float),
                                               iBias_2.data(), &err)); //change this to reflect project and same #
    //Layer3
    OCL_CHECK(err, cl::Buffer iWbuffer_3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 64 * 2 * sizeof(float),
                                               iWeights_3.data(), &err)); //change this to reflect project and same #
    OCL_CHECK(err, cl::Buffer iBbuffer_3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 2 * sizeof(float),
                                               iBias_3.data(), &err)); //change this to reflect project and same #
    //Output
    OCL_CHECK(err, cl::Buffer buffer_e(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, vector_size_bytes,
                                           result_krnl.data(), &err)); //related to spectrumSensing top so output is 2

    range.end();

    events.mark("Beffers created");

    range.start("Phase 4", "Setting up arguments and running kernel");

    // Set the kernel arguments

    OCL_CHECK(err, err = krnl.setArg(0, input_buff));
    //L0
    OCL_CHECK(err, err = krnl.setArg(1, iWbuffer_0));
    OCL_CHECK(err, err = krnl.setArg(2, iBbuffer_0));
    //L1
    OCL_CHECK(err, err = krnl.setArg(3, iWbuffer_1));
    OCL_CHECK(err, err = krnl.setArg(4, iBbuffer_1));
    //L2
    OCL_CHECK(err, err = krnl.setArg(5, iWbuffer_2));
    OCL_CHECK(err, err = krnl.setArg(6, iBbuffer_2));
    //L3
    OCL_CHECK(err, err = krnl.setArg(7, iWbuffer_3));
    OCL_CHECK(err, err = krnl.setArg(8, iBbuffer_3));

    OCL_CHECK(err, err = krnl.setArg(9, buffer_e));
    //OCL_CHECK(err, err = krnl.setArg(2, vector_length));

    // Copy input vectors to memory
    //OCL_CHECK(err, err = q.enqueueMigrateMemObjects({input_buff, iWbuffer_0}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({
        input_buff,
        iWbuffer_0, iBbuffer_0,
        iWbuffer_1, iBbuffer_1,
        iWbuffer_2, iBbuffer_2,
        iWbuffer_3, iBbuffer_3
    }, 0));

    // Launch the kernel and get profile data (stop-start)
    uint64_t nstimestart, nstimeend;
    cl::Event event;
    OCL_CHECK(err, err = q.enqueueTask(krnl, nullptr, &event));
    OCL_CHECK(err, err = q.finish());
    OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
    OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
    auto duration_nanosec = nstimeend - nstimestart;
    std::cout << " **** Duration returned by profile API is " << (duration_nanosec * (1.0e-6)) << " ms **** "
              << std::endl;

    // Copy result to local buffer
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_e}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q.finish());

    range.end();

    events.mark("Kernels finished");

    range.start("Phase 5", "Verification");

    writeFP<float>(base_dir + "/rv_data.csv",&result_krnl[0],1,2);

    // Compare the results of the kernel to the simulation
    range.end();
    events.mark("Verification done");

    //Call spectrumSensing here
    spectrumSensing_top(&input[0],
    		&iWeights_0[0], &iBias_0[0],
    		&iWeights_1[0], &iBias_1[0],
			&iWeights_2[0], &iBias_2[0],
			&iWeights_3[0], &iBias_3[0],
			&oData[0]);

    for(int i = 0; i< 2; i++){
    	std::cout << "Result: " << i+1 << std::endl;
    	std::cout << "Difference: " << oData[i] - result_krnl[i] << std::endl;
    	std::cout << "Software result: " << oData[i] << std::endl;
    	std::cout << "Kernel output: " <<  result_krnl[i] << std::endl;
    	std::cout << "========================" << std::endl;
    }

    std::cout << "TEST PASSED" << std::endl;
    return 0;
}
