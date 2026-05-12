// GPRat
#include "target.hpp"
#include "gpu/sycl/sycl_gp_uncertainty.hpp"
#include "gpu/sycl/sycl_utils.hpp"

// oneMath
#include <oneapi/math.hpp>

namespace gprat::sycl_backend
{

double *diag_posterior(double *A, double *B, std::size_t M)
{
    sycl::queue queue(sycl::gpu_selector_v);

    double *tile = sycl::malloc_device<double>(M, queue);

    // tile = 1.0*A + (-1.0)*B
    oneapi::math::blas::column_major::omatadd(
        queue,
        oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans,
        1,
        static_cast<int64_t>(M),
        1.0,
        A,
        1,
        -1.0,
        B,
        1,
        tile,
        1
    );

    queue.wait();

    return tile;
}

double *diag_tile(
    double *A, 
    std::size_t M
)
{
    //sycl::queue queue = sycl_device.next_queue();
    sycl::queue queue(sycl::gpu_selector_v);

    double *diag_tile = sycl::malloc_device<double>(M, queue);

    oneapi::math::blas::column_major::omatcopy(
        queue,
        oneapi::math::transpose::nontrans,
        1,                                  
        static_cast<int64_t>(M),                                  
        1.0,
        A,
        static_cast<int64_t>(M) + 1,
        diag_tile,
        1 
    );

    queue.wait();

    return diag_tile;
}

}  // end of namespace gprat::sycl_backend
