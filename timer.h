#pragma once

#include <chrono>
#include <cstring>
#include <iostream>

struct Timer {
    using clock_t = std::chrono::high_resolution_clock;
    using time_point_t = std::chrono::time_point<clock_t>;
    using elapsed_time_t = std::chrono::duration<double, std::milli>;

    time_point_t mStartTime;
    time_point_t mStopTime;

    void start() {
        mStartTime = clock_t::now();
    }

    double stop(const std::string& msg) {
        mStopTime = clock_t::now();
        std::chrono::duration<double, std::milli> elapsedTime = mStopTime - mStartTime;
        std::cout << "[" << msg << elapsedTime.count() << "ms]" << std::endl;
        return elapsedTime.count();
    }
};
