#ifndef CUBLAS_ADAPTER_H_INCLUDED
#define CUBLAS_ADAPTER_H_INCLUDED

#include <hpx/local/future.hpp>
#include <hpx/modules/async_cuda.hpp>

//C = C - A * B^T
template <typename T>
hpx::shared_future<host_buffer_t<T>> gemm_cublas(hpx::cuda::experimental::cublas_executor& cublas,
                                               hpx::shared_future<host_buffer_t<T>> A,
                                               hpx::shared_future<host_buffer_t<T>> B,
                                               hpx::shared_future<host_buffer_t<T>> C,
                                               std::size_t N)
{
  // GEMM constants
  const T alpha = -1.0f;
  const T beta = 1.0f;
  std::size_t size = N * N;
  // allocate device memory
  cuda_device_buffer_t<T> d_A(size);
  cuda_device_buffer_t<T> d_B(size);
  cuda_device_buffer_t<T> d_C(size);
  // copy data from host to device when ready
  host_buffer_t<T> h_A = A.get();
  hpx::apply(cublas, cudaMemcpyAsync, d_A.device_side_buffer, h_A.data(), size * sizeof(T), cudaMemcpyHostToDevice);
  host_buffer_t<T> h_B = B.get();
  hpx::apply(cublas, cudaMemcpyAsync, d_B.device_side_buffer, h_B.data(), size * sizeof(T), cudaMemcpyHostToDevice);
  host_buffer_t<T> h_C = C.get();
  hpx::apply(cublas, cudaMemcpyAsync, d_C.device_side_buffer, h_C.data(), size * sizeof(T), cudaMemcpyHostToDevice);
  // compute GEMM (C = C - A * B^T) on device
  // note cublas uses column major ordering : (A * B^T)^T = B * A^T
  hpx::apply(cublas, cublasSgemm, CUBLAS_OP_T, CUBLAS_OP_N, N, N, N, &alpha, d_B.device_side_buffer, N, d_A.device_side_buffer, N, &beta, d_C.device_side_buffer, N);
  // copy the result back to the host
  auto copy_device_to_host = hpx::async(cublas, cudaMemcpyAsync, h_C.data(), d_C.device_side_buffer, size * sizeof(T), cudaMemcpyDeviceToHost);
  copy_device_to_host.get();
  // return future
  return hpx::make_ready_future(h_C);
}

//  A = A - B * B^T
template <typename T>
hpx::shared_future<host_buffer_t<T>> syrk_cublas(hpx::cuda::experimental::cublas_executor& cublas,
                                               hpx::shared_future<host_buffer_t<T>> A,
                                               hpx::shared_future<host_buffer_t<T>> B,
                                               std::size_t N)
{
  // GEMM constants
  const T alpha = -1.0f;
  const T beta = 1.0f;
  std::size_t size = N * N;
  // allocate device memory
  cuda_device_buffer_t<T> d_A(size);
  cuda_device_buffer_t<T> d_B(size);
  // copy data from host to device when ready
  host_buffer_t<T> h_A = A.get();
  hpx::apply(cublas, cudaMemcpyAsync, d_A.device_side_buffer, h_A.data(), size * sizeof(T), cudaMemcpyHostToDevice);
  host_buffer_t<T> h_B = B.get();
  hpx::apply(cublas, cudaMemcpyAsync, d_B.device_side_buffer, h_B.data(), size * sizeof(T), cudaMemcpyHostToDevice);
  // compute SYRK (A = A - B * B^T) on device
  hpx::apply(cublas, cublasSgemm, CUBLAS_OP_T, CUBLAS_OP_N, N, N, N, &alpha, d_B.device_side_buffer, N, d_B.device_side_buffer, N, &beta, d_A.device_side_buffer, N);
  // copy the result back to the host
  auto copy_device_to_host = hpx::async(cublas, cudaMemcpyAsync, h_A.data(), d_A.device_side_buffer, size * sizeof(T), cudaMemcpyDeviceToHost);
  copy_device_to_host.get();
  // return future
  return hpx::make_ready_future(h_A);
}
#endif
