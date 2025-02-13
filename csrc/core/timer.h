#pragma once

#include "common.h"
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <unordered_map>

class CudaTimer {
public:
    explicit CudaTimer(cudaStream_t stream = 0, std::string tag = "")
        : stream_(stream), tag_(tag), is_recording_(false) {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~CudaTimer() {
        CUDA_CHECK_NOTHROW(cudaEventDestroy(start_));
        CUDA_CHECK_NOTHROW(cudaEventDestroy(stop_));
    }

    void start() {
        if (!is_recording_) {
            CUDA_CHECK(cudaEventRecord(start_, stream_));
            is_recording_ = true;
        }
    }

    float stop() {
        if (is_recording_) {
            CUDA_CHECK(cudaEventRecord(stop_, stream_));
            CUDA_CHECK(cudaEventSynchronize(stop_));
            is_recording_ = false;

            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
            return ms;
        }
        return 0.0f;
    }

    static void add_record(const std::string& tag, float ms) {
        get_records().emplace_back(tag, ms);
    }

    static void print_all() {
        const auto& records = get_records();
        if (records.empty()) return;

        std::cout << "\n=== CUDA Kernel Timing Summary ===" << std::endl;
        std::cout << std::setw(40) << std::left << "[Kernel Tag]"
                  << std::setw(12) << "Time(ms)" << std::endl;

        std::unordered_map<std::string, std::pair<float, int>> stats;
        for (const auto& [tag, ms] : records) {
            stats[tag].first += ms;
            stats[tag].second++;
        }

        for (const auto& [tag, data] : stats) {
            std::cout << std::setw(40) << std::left << tag
                      << std::fixed << std::setprecision(3)
                      << std::setw(12) << data.first / data.second
                      << " (avg over " << data.second << " runs)\n";
        }
    }

private:
    cudaEvent_t start_, stop_;
    cudaStream_t stream_;
    std::string tag_;
    bool is_recording_;

    // 线程局部存储保证多线程安全
    static std::vector<std::pair<std::string, float>>& get_records() {
        thread_local static std::vector<std::pair<std::string, float>> records;
        return records;
    }
};