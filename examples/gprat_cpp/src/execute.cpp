#include "gprat_c.hpp"
#include "utils_c.hpp"
#include <chrono>
#include <fstream>
#include <iostream>

namespace gprat::example
{
    struct Runtimes
    {
        std::chrono::duration<double> init;
        std::chrono::duration<double> cholesky;
        std::chrono::duration<double> opt;
        std::chrono::duration<double> pred_uncer;
        std::chrono::duration<double> pred_full_cov;
        std::chrono::duration<double> pred;
    };

    constexpr std::size_t START = 512;
    constexpr std::size_t END = 1024;
    constexpr std::size_t STEP = 2;
    constexpr std::size_t LOOP = 2;
    constexpr std::size_t OPT_ITER = 1;

    constexpr std::size_t N_CORES = 4;
    constexpr std::size_t n_tiles = 16;
    constexpr std::size_t n_reg = 8;
    constexpr int n_test = 1024;

    // SYCL test settings
    constexpr int device_id = 0;
    constexpr int n_queues = 8;

    // Save parameters and times to a .txt file with a header
    void append_to_output_file(
        std::string &target,
        int &n_train,
        std::size_t &core,
        std::size_t &l,
        Runtimes &runtimes,
        std::chrono::nanoseconds &total_time
    )
    {
        
        std::ofstream outfile("../output.csv", std::ios::app);  // Append mode
        if (outfile.tellp() == 0)
        {
            // If file is empty, write the header
            outfile << "Target,Cores,N_train,N_test,N_tiles,N_regressor,Opt_iter,Total_time,runtimes.init,Cholesky_"
                        "time,runtimes.opt,runtimes.pred_uncer,Pred_Full_time,runtimes.pred,N_loop\n";
        }
        outfile << target << "," << core << "," << n_train << "," << gprat::example::n_test << "," << gprat::example::n_tiles << "," << gprat::example::n_reg
                << "," << gprat::example::OPT_ITER << "," << total_time.count() << "," << runtimes.init.count() << ","
                << runtimes.cholesky.count() << "," << runtimes.opt.count() << "," << runtimes.pred_uncer.count() << ","
                << runtimes.pred_full_cov.count() << "," << runtimes.pred.count() << "," << l << "\n";
        outfile.close();
    }

    void example_cpu(
        Runtimes &runtimes,
        std::string &target,
        std::pair<int, int> &result,
        gprat::GP_data &training_input,
        gprat::GP_data &training_output,
        gprat::GP_data &test_input,
        const int tile_size,
        std::vector<bool> trainable
    )
    {
        gprat_hyper::AdamParams hpar = { 0.1, 0.9, 0.999, 1e-8, gprat::example::OPT_ITER };

        target = "cpu";

        auto start_init = std::chrono::high_resolution_clock::now();
        gprat::GP gp_cpu(training_input.data,
                            training_output.data,
                            gprat::example::n_tiles,
                            tile_size,
                            gprat::example::n_reg,
                            { 1.0, 1.0, 0.1 },
                            trainable);
        auto end_init = std::chrono::high_resolution_clock::now();
        runtimes.init = end_init - start_init;

        auto start_cholesky = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<double>> cholesky_cpu = gp_cpu.cholesky();
        auto end_cholesky = std::chrono::high_resolution_clock::now();
        runtimes.cholesky = end_cholesky - start_cholesky;

        auto start_opt = std::chrono::high_resolution_clock::now();
        std::vector<double> losses = gp_cpu.optimize(hpar);
        auto end_opt = std::chrono::high_resolution_clock::now();
        runtimes.opt = end_opt - start_opt;

        auto start_pred_uncer = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<double>> sum_cpu =
            gp_cpu.predict_with_uncertainty(test_input.data, result.first, result.second);
        auto end_pred_uncer = std::chrono::high_resolution_clock::now();
        runtimes.pred_uncer = end_pred_uncer - start_pred_uncer;

        auto start_pred_full_cov = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<double>> full_cpu =
            gp_cpu.predict_with_full_cov(test_input.data, result.first, result.second);
        auto end_pred_full_cov = std::chrono::high_resolution_clock::now();
        runtimes.pred_full_cov = end_pred_full_cov - start_pred_full_cov;

        auto start_pred = std::chrono::high_resolution_clock::now();
        std::vector<double> pred_cpu = gp_cpu.predict(test_input.data, result.first, result.second);
        auto end_pred = std::chrono::high_resolution_clock::now();
        runtimes.pred = end_pred - start_pred;
    }

    void example_cuda_gpu(
        Runtimes &runtimes,
        std::string &target,
        std::pair<int, int> &result,
        gprat::GP_data &training_input,
        gprat::GP_data &training_output,
        gprat::GP_data &test_input,
        const int tile_size,
        std::vector<bool> trainable
    )
    {
        target = "cuda";

        auto start_init = std::chrono::high_resolution_clock::now();
        gprat::GP gp_gpu(
            training_input.data,
            training_output.data,
            gprat::example::n_tiles,
            tile_size,
            gprat::example::n_reg,
            { 1.0, 1.0, 0.1 },
            trainable,
            0,
            2);
        auto end_init = std::chrono::high_resolution_clock::now();
        runtimes.init = end_init - start_init;

        auto start_cholesky = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<double>> cholesky_gpu = gp_gpu.cholesky();
        auto end_cholesky = std::chrono::high_resolution_clock::now();
        runtimes.cholesky = end_cholesky - start_cholesky;

        // NOTE: optimization is not implemented for GPU
        runtimes.opt = std::chrono::seconds(-1);

        auto start_pred_uncer = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<double>> sum_gpu =
            gp_gpu.predict_with_uncertainty(test_input.data, result.first, result.second);
        auto end_pred_uncer = std::chrono::high_resolution_clock::now();
        runtimes.pred_uncer = end_pred_uncer - start_pred_uncer;

        auto start_pred_full_cov = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<double>> full_gpu =
            gp_gpu.predict_with_full_cov(test_input.data, result.first, result.second);
        auto end_pred_full_cov = std::chrono::high_resolution_clock::now();
        runtimes.pred_full_cov = end_pred_full_cov - start_pred_full_cov;

        auto start_pred = std::chrono::high_resolution_clock::now();
        std::vector<double> pred_gpu = gp_gpu.predict(test_input.data, result.first, result.second);
        auto end_pred = std::chrono::high_resolution_clock::now();
        runtimes.pred = end_pred - start_pred;
    }

    void example_sycl(
        Runtimes &runtimes,
        std::string &target,
        std::pair<int, int> &result,
        gprat::GP_data &training_input,
        gprat::GP_data &training_output,
        gprat::GP_data &test_input,
        const int tile_size,
        std::vector<bool> trainable
    )
    {
        target = "sycl";

        auto start_init = std::chrono::high_resolution_clock::now();
        gprat::GP gp_sycl(
                    training_input.data,
                    training_output.data,
                    n_tiles,
                    tile_size,
                    n_reg,
                    std::vector<double>{1.0, 1.0, 0.1},
                    trainable,
                    device_id, 
                    n_queues
        );

        auto end_init = std::chrono::high_resolution_clock::now();
        runtimes.init = end_init - start_init;

        auto start_cholesky = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<double>> cholesky_sycl = gp_sycl.cholesky();
        auto end_cholesky = std::chrono::high_resolution_clock::now();
        runtimes.cholesky = end_cholesky - start_cholesky;

        // NOTE: optimization is not implemented for GPU
        runtimes.opt = std::chrono::seconds(-1);

        auto start_pred_uncer = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<double>> sum_sycl =
            gp_sycl.predict_with_uncertainty(test_input.data, result.first, result.second);
        auto end_pred_uncer = std::chrono::high_resolution_clock::now();
        runtimes.pred_uncer = end_pred_uncer - start_pred_uncer;

        auto start_pred_full_cov = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<double>> full_sycl =
            gp_sycl.predict_with_full_cov(test_input.data, result.first, result.second);
        auto end_pred_full_cov = std::chrono::high_resolution_clock::now();
        runtimes.pred_full_cov = end_pred_full_cov - start_pred_full_cov;

        auto start_pred = std::chrono::high_resolution_clock::now();
        std::vector<double> pred_sycl = gp_sycl.predict(test_input.data, result.first, result.second);
        auto end_pred = std::chrono::high_resolution_clock::now();
        runtimes.pred = end_pred - start_pred;
    }

} // ! namespace gprat::example

int main(int argc, char *argv[])
{
    std::string train_path = "../../../../data/data_19/input_data.txt";
    std::string out_path = "../../../../data/data_19/output_data.txt";
    std::string test_path = "../../../../data/data_19/test_input.txt";

    bool use_gpu =
        utils::compiled_with_cuda() && gprat::gpu_count() > 0 && argc > 1 && std::strcmp(argv[1], "--use_cuda") == 0;

    bool use_sycl =
        utils::compiled_with_sycl() && gprat::gpu_count() > 0 && argc > 1 && std::strcmp(argv[1], "--use_sycl") == 0;

    for (std::size_t core = 2; core <= gprat::example::N_CORES; core = core * 2)
    {
        // Create new argc and argv to include the --hpx:threads argument
        std::vector<std::string> args(argv, argv + argc);
        if (use_gpu || use_sycl) { args.erase(args.begin() + 1); }
        args.push_back("--hpx:threads=" + std::to_string(core));

        // Convert the arguments to char* array
        std::vector<char *> cstr_args;
        for (auto &arg : args) { cstr_args.push_back(const_cast<char *>(arg.c_str())); }

        int new_argc = static_cast<int>(cstr_args.size());
        char **new_argv = cstr_args.data();

        utils::start_hpx_runtime(new_argc, new_argv);

        for (std::size_t start = gprat::example::START; start <= gprat::example::END; start = start * gprat::example::STEP)
        {
            int n_train = static_cast<int>(start);

            for (std::size_t l = 0; l < gprat::example::LOOP; l++)
            {
                int tile_size = utils::compute_train_tile_size(n_train, gprat::example::n_tiles);
                auto result = utils::compute_test_tiles(gprat::example::n_test, gprat::example::n_tiles, tile_size);

                gprat::GP_data training_input(train_path, n_train, gprat::example::n_reg);
                gprat::GP_data training_output(out_path, n_train, gprat::example::n_reg);
                gprat::GP_data test_input(test_path, gprat::example::n_test, gprat::example::n_reg);

                gprat::example::Runtimes runtimes;
                std::vector<bool> trainable = { true, true, true };
                std::string target;

                auto start_total = std::chrono::high_resolution_clock::now();

                if (!use_gpu && !use_sycl)
                {
                    gprat::example::example_cpu(
                        runtimes,
                        target,
                        result,
                        training_input,
                        training_output,
                        test_input,
                        tile_size,
                        trainable 
                    );
                }
                else if (use_gpu)
                {
                    gprat::example::example_cuda_gpu(
                        runtimes,
                        target,
                        result,
                        training_input,
                        training_output,
                        test_input,
                        tile_size,
                        trainable 
                    );
                }
                else if(use_sycl)
                {
                    gprat::example::example_sycl(
                        runtimes,
                        target,
                        result,
                        training_input,
                        training_output,
                        test_input,
                        tile_size,
                        trainable 
                    );   
                }

                auto end_total = std::chrono::high_resolution_clock::now();
                auto total_time = end_total - start_total;

                gprat::example::append_to_output_file(target, n_train, core, l, runtimes, total_time);
            }
        }

        utils::stop_hpx_runtime();
    }

    return 0;
}
