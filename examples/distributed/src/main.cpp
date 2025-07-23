#include "gprat/cpu/gp_algorithms.hpp"
#include "gprat/kernels.hpp"

#include "../../test/src/test_data.hpp"
#include "distributed_blas.hpp"
#include "distributed_cholesky.hpp"
#include "distributed_tile.hpp"
#include <fstream>
#include <hpx/compute.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/hpx_init_params.hpp>
#include <span>

// This is a standalone test, so including this directly is fine.
// Better than having the whole project depend on compiled Boost.Json!

#include "gprat/gprat.hpp"
#include "gprat/utils.hpp"

#include <boost/json/src.hpp>

GPRAT_REGISTER_TILED_DATASET(double, double);

GPRAT_NS_BEGIN

hpx::future<tile_handle<double>> gen_tile_covariance_distributed(
    tile_handle<double> tile,
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const SEKParams &sek_params,
    std::span<const double> input)
{
    return tile.set_async(cpu::gen_tile_covariance(row, col, N, n_regressors, sek_params, input));
}

HPX_DEFINE_PLAIN_DIRECT_ACTION(gen_tile_covariance_distributed);
GPRAT_DECLARE_PLAIN_ACTION_FOR(&cpu::gen_tile_covariance,
                               gen_tile_covariance_distributed_action,
                               "gen_tile_covariance");

template <typename Tiles, typename Scheduler = tiled_scheduler_local>
void right_looking_cholesky_tiled(Scheduler &sched, Tiles &ft_tiles, std::size_t N, std::size_t n_tiles)
{
    for (std::size_t k = 0; k < n_tiles; k++)
    {
        // POTRF: Compute Cholesky factor L
        ft_tiles[k * n_tiles + k] = detail::named_dataflow<potrf>(
            sched, cholesky_POTRF(sched, k), "cholesky_tiled", ft_tiles[k * n_tiles + k], N);
        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            // TRSM:  Solve X * L^T = A
            ft_tiles[m * n_tiles + k] = detail::named_dataflow<trsm>(
                sched,
                cholesky_TRSM(sched, k, m),
                "cholesky_tiled",
                ft_tiles[k * n_tiles + k],
                ft_tiles[m * n_tiles + k],
                N,
                N,
                Blas_trans,
                Blas_right);
        }
        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            // SYRK:  A = A - B * B^T
            ft_tiles[m * n_tiles + m] = detail::named_dataflow<syrk>(
                sched,
                cholesky_SYRK(sched, m),
                "cholesky_tiled",
                ft_tiles[m * n_tiles + m],
                ft_tiles[m * n_tiles + k],
                N);
            for (std::size_t n = k + 1; n < m; n++)
            {
                // GEMM: C = C - A * B^T
                ft_tiles[m * n_tiles + n] = detail::named_dataflow<gemm>(
                    sched,
                    cholesky_GEMM(sched, k, m, n),
                    "cholesky_tiled",
                    ft_tiles[m * n_tiles + k],
                    ft_tiles[n * n_tiles + k],
                    ft_tiles[m * n_tiles + n],
                    N,
                    N,
                    N,
                    Blas_no_trans,
                    Blas_trans);
            }
        }
    }
}

template <typename Scheduler = tiled_scheduler_local>
std::vector<mutable_tile_data<double>>
cholesky_hpx(Scheduler &sched,
             std::span<const double> training_input,
             const SEKParams &sek_params,
             std::size_t n_tiles,
             std::size_t n_tile_size,
             std::size_t n_regressors)
{
    auto tiles = make_cholesky_dataset<double>(sched, n_tiles);  // Tiled covariance matrix

    for (std::size_t row = 0; row < n_tiles; row++)
    {
        for (std::size_t col = 0; col <= row; col++)
        {
            tiles[row * n_tiles + col] = detail::named_dataflow<cpu::gen_tile_covariance>(
                sched,
                cholesky_tile(sched, row, col),
                "cholesky init",
                tiles[row * n_tiles + col],
                row,
                col,
                n_tile_size,
                n_regressors,
                sek_params,
                training_input);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled(sched, tiles, n_tile_size, n_tiles);

    ///////////////////////////////////////////////////////////////////////////
    // Synchronize
    std::vector<mutable_tile_data<double>> result(n_tiles * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            result[i * n_tiles + j] = tiles[i * n_tiles + j].get();
        }
    }
    // hpx::get_runtime_distributed().evaluate_active_counters(false, "POST cholesky");
    return result;
}

gprat_results load_test_data_results(const std::string &filename)
{
    std::ifstream ifs(filename);
    if (!ifs.fail())
    {
        using iterator_type = std::istreambuf_iterator<char>;
        const std::string content(iterator_type{ ifs }, iterator_type{});
        return boost::json::value_to<gprat_results>(boost::json::parse(content));
    }
    throw std::runtime_error("Failed to load " + filename);
}

void validate_two_dim_result(const std::vector<std::vector<double>> &expected,
                             const std::vector<mutable_tile_data<double>> &actual)
{
    if (expected.size() != actual.size())
    {
        throw std::runtime_error("expected.size() != actual.size()");
    }

    constexpr double margin = 0.00001;
    bool is_valid = true;
    for (std::size_t i = 0; i < expected.size(); i++)
    {
        if (expected[i].size() != actual[i].size())
        {
            throw std::runtime_error("expected[i].size() != actual[i].size(): i = " + std::to_string(i));
        }

        const std::span<const double> actual_data = actual[i];
        for (std::size_t j = 0; j < expected[i].size(); j++)
        {
            const auto &expected_value = expected[i][j];
            const auto &actual_value = actual_data[j];

            // XXX: no std::abs(expected - actual) due to infinity
            const bool is_in_range =
                (expected_value + margin >= actual_value) && (actual_value + margin >= expected_value);
            if (!is_in_range)
            {
                std::cerr << "MISMATCH at " << i << " " << j << " " << expected_value << " !~= " << actual_value
                          << std::endl;
                is_valid = false;
            }
        }
    }

    if (!is_valid)
    {
        throw std::runtime_error("Invalid results (see stderr for details)");
    }
}

void run(hpx::program_options::variables_map &vm)
{
    /////////////////////
    /////// configuration
    std::size_t START = vm["start"].as<std::size_t>();
    std::size_t END = vm["end"].as<std::size_t>();
    std::size_t STEP = vm["step"].as<std::size_t>();
    std::size_t LOOP = vm["loop"].as<std::size_t>();
    const int OPT_ITER = vm["opt_iter"].as<int>();

    int n_test = 1024;
    const std::size_t n_tiles = vm["tiles"].as<std::size_t>();
    const std::size_t n_reg = vm["regressors"].as<std::size_t>();

    const auto &train_path = vm["train_x_path"].as<std::string>();
    const auto &out_path = vm["train_y_path"].as<std::string>();
    const auto &test_path = vm["test_path"].as<std::string>();

    std::optional<gprat_results> test_results;
    // XXX: cannot use contains() because it's not exported by HPX program_options
    // ReSharper disable once CppUseAssociativeContains
    if (vm.find("test_results_path") != vm.end())
    {
        test_results = load_test_data_results(vm["test_results_path"].as<std::string>());
        std::cerr << "We have comparison data!" << std::endl;
    }

    scheduler::tiled_cholesky_scheduler_paap12 scheduler;

    for (std::size_t start = START; start <= END; start = start * STEP)
    {
        int n_train = static_cast<int>(start);
        for (std::size_t l = 0; l < LOOP; l++)
        {
            hpx::chrono::high_resolution_timer total_timer;

            // Compute tile sizes and number of predict tiles
            int tile_size = compute_train_tile_size(n_train, n_tiles);
            auto result = compute_test_tiles(n_test, n_tiles, tile_size);
            /////////////////////
            ///// hyperparams
            AdamParams hpar = { 0.1, 0.9, 0.999, 1e-8, OPT_ITER };

            /////////////////////
            ////// data loading
            GP_data training_input(train_path, n_train, n_reg);
            GP_data training_output(out_path, n_train, n_reg);
            GP_data test_input(test_path, n_test, n_reg);

            /////////////////////
            ///// GP
            hpx::chrono::high_resolution_timer init_timer;
            std::vector<bool> trainable = { true, true, true };
            GP gp(training_input.data, training_output.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, trainable);
            const auto init_time = init_timer.elapsed();

            // Measure the time taken to execute gp.cholesky();
            auto start_cholesky = std::chrono::high_resolution_clock::now();

            hpx::chrono::high_resolution_timer cholesky_timer;
            const auto cholesky =
                cholesky_hpx(scheduler, training_input.data, { 1.0, 1.0, 0.1 }, n_tiles, tile_size, n_reg);
            const auto cholesky_time = cholesky_timer.elapsed();

            // Save parameters and times to a .txt file with a header
            std::ofstream outfile(vm["timings_csv"].as<std::string>(), std::ios::app);
            if (outfile.tellp() == 0)
            {
                // If file is empty, write the header
                outfile << "Cores,N_train,N_test,N_tiles,N_regressor,Opt_iter,Total_time,Init_time,Cholesky_time,"
                           "Opt_time,Pred_Uncer_time,Pred_Full_time,Pred_time,N_loop\n";
            }
            outfile << hpx::get_locality_id() << "," << n_train << "," << n_test << "," << n_tiles << "," << n_reg
                    << "," << OPT_ITER << "," << total_timer.elapsed() << "," << init_time << "," << cholesky_time
                    << "," << 0 << "," << 0 << "," << 0 << "," << 0 << "," << l << "\n";
            outfile.close();

            if (test_results)
            {
                std::cerr << "Validating results..." << std::endl;
                validate_two_dim_result(test_results->choleksy, cholesky);
            }
        }
    }
    std::cerr << "DONE!" << std::endl;
}

GPRAT_NS_END

HPX_REGISTER_ACTION(GPRAT_NS::gen_tile_covariance_distributed_action);

int hpx_main(hpx::program_options::variables_map &vm)
{
    std::cerr << "OS Threads: " << hpx::get_os_thread_count() << std::endl;
    std::cerr << "All localities: " << hpx::get_num_localities().get() << std::endl;
    std::cerr << "Root locality: " << hpx::find_root_locality() << std::endl;
    std::cerr << "This locality: " << hpx::find_here() << std::endl;
    std::cerr << "Remote localities: " << hpx::find_remote_localities().size() << std::endl;

    auto numa_domains = hpx::compute::host::numa_domains();
    try
    {
        GPRAT_NS::run(vm);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }
    return hpx::finalize();
}

int main(int argc, char *argv[])
{
    hpx::register_startup_function(&GPRAT_NS::register_performance_counters);
    hpx::register_startup_function(&GPRAT_NS::register_distributed_tile_counters);
    hpx::register_startup_function(&GPRAT_NS::register_distributed_blas_counters);

    namespace po = hpx::program_options;
    po::options_description desc("Allowed options");

    // clang-format off
    desc.add_options()
        ("help", "produce help message")
        ("train_x_path", po::value<std::string>()->default_value("data/data_1024/training_input.txt"), "training data (x)")
        ("train_y_path", po::value<std::string>()->default_value("data/data_1024/training_output.txt"), "training data (y)")
        ("test_path", po::value<std::string>()->default_value("data/data_1024/test_input.txt"), "test data")
        ("test_results_path", po::value<std::string>()->default_value("data/data_1024/output.json"), "test data results to validate results with")
        ("timings_csv", po::value<std::string>()->default_value("timings.csv"), "output timing reports")
        ("tiles", po::value<std::size_t>()->default_value(16), "tiles per dimension")
        ("regressors", po::value<std::size_t>()->default_value(8), "num regressors")
        ("start", po::value<std::size_t>()->default_value(128), "Starting number of training samples")
        ("end", po::value<std::size_t>()->default_value(128), "End number of training samples")
        ("step", po::value<std::size_t>()->default_value(2), "Increment of training samples")
        ("loop", po::value<std::size_t>()->default_value(1), "Number of iterations to be performed for each number of training samples")
        ("opt_iter", po::value<int>()->default_value(3), "Number of optimization iterations*/")
    ;
    // clang-format on

    hpx::init_params init_args;
    init_args.desc_cmdline = desc;
    // If example requires to run hpx_main on all localities
    // std::vector<std::string> const cfg = {"hpx.run_hpx_main!=1"};
    // init_args.cfg = cfg;
    // Run HPX main
    return hpx::init(argc, argv, init_args);
}
