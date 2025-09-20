#include "distributed_blas.hpp"
#include "distributed_cholesky.hpp"
#include "distributed_tile.hpp"

#include "gprat/cpu/gp_algorithms.hpp"
#include "gprat/cpu/gp_functions.hpp"
#include "gprat/gprat.hpp"
#include "gprat/kernels.hpp"
#include "gprat/performance_counters.hpp"
#include "gprat/scheduler.hpp"
#include "gprat/utils.hpp"

#include "../../test/src/test_data.hpp"
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fstream>
#include <hpx/compute.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/hpx_init_params.hpp>
#include <iostream>
#include <span>

// This is a standalone test, so including this directly is fine.
// Better than having the whole project depend on compiled Boost.Json!
#include <boost/json/src.hpp>

GPRAT_REGISTER_TILED_DATASET(double, double);

GPRAT_NS_BEGIN

template <typename T, typename Mapper>
tiled_dataset<T> make_tiled_dataset(const tiled_scheduler_distributed &sched, std::size_t num_tiles, Mapper &&mapper)
{
    const auto num_localities = sched.localities_.size();
    std::vector<std::pair<hpx::id_type, std::size_t>> targets;
    targets.reserve(num_localities);

    for (std::size_t i = 0; i < num_localities; ++i)
    {
        targets.emplace_back(sched.localities_[i], 0);
    }

    for (std::size_t i = 0; i < num_tiles; i++)
    {
        ++targets[mapper(i) % num_localities].second;
    }

    return create_tiled_dataset<T>(targets, num_tiles);
}

hpx::future<tile_handle<double>> gen_tile_covariance_distributed(
    const tile_handle<double> &tile,
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const SEKParams &sek_params,
    const std::vector<double> &input);
HPX_DEFINE_PLAIN_DIRECT_ACTION(gen_tile_covariance_distributed);
GPRAT_DECLARE_PLAIN_ACTION_FOR(&cpu::gen_tile_covariance,
                               gen_tile_covariance_distributed_action,
                               "gen_tile_covariance");
GPRAT_DEFINE_PLAIN_ACTION_FOR(&cpu::gen_tile_covariance);

hpx::future<tile_handle<double>> gen_tile_covariance_distributed(
    const tile_handle<double> &tile,
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const SEKParams &sek_params,
    const std::vector<double> &input)
{
    return tile.set_async(cpu::gen_tile_covariance(row, col, N, n_regressors, sek_params, input));
}

hpx::future<tile_handle<double>> gen_tile_covariance_with_distance_distributed(
    const tile_handle<double> &tile,
    std::size_t row,
    std::size_t col,
    std::size_t N,
    const SEKParams &sek_params,
    const const_tile_data<double> &distance);

HPX_DEFINE_PLAIN_DIRECT_ACTION(gen_tile_covariance_with_distance_distributed);
GPRAT_DECLARE_PLAIN_ACTION_FOR(&cpu::gen_tile_covariance_with_distance,
                               gen_tile_covariance_with_distance_distributed_action,
                               "gen_tile_covariance_with_distance");
GPRAT_DEFINE_PLAIN_ACTION_FOR(&cpu::gen_tile_covariance_with_distance);

hpx::future<tile_handle<double>> gen_tile_covariance_with_distance_distributed(
    const tile_handle<double> &tile,
    std::size_t row,
    std::size_t col,
    std::size_t N,
    const SEKParams &sek_params,
    const const_tile_data<double> &distance)
{
    return tile.set_async(cpu::gen_tile_covariance_with_distance(row, col, N, sek_params, distance));
}

hpx::future<tile_handle<double>> gen_tile_prior_covariance_distributed(
    const tile_handle<double> &tile,
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const SEKParams &sek_params,
    const std::vector<double> &input);
HPX_DEFINE_PLAIN_DIRECT_ACTION(gen_tile_prior_covariance_distributed);
GPRAT_DECLARE_PLAIN_ACTION_FOR(&cpu::gen_tile_prior_covariance,
                               gen_tile_prior_covariance_distributed_action,
                               "gen_tile_prior_covariance");
GPRAT_DEFINE_PLAIN_ACTION_FOR(&cpu::gen_tile_prior_covariance);

hpx::future<tile_handle<double>> gen_tile_prior_covariance_distributed(
    const tile_handle<double> &tile,
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const SEKParams &sek_params,
    const std::vector<double> &input)
{
    return tile.set_async(cpu::gen_tile_prior_covariance(row, col, N, n_regressors, sek_params, input));
}

hpx::future<tile_handle<double>> gen_tile_full_prior_covariance_distributed(
    const tile_handle<double> &tile,
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const SEKParams &sek_params,
    const std::vector<double> &input);
HPX_DEFINE_PLAIN_DIRECT_ACTION(gen_tile_full_prior_covariance_distributed);
GPRAT_DECLARE_PLAIN_ACTION_FOR(&cpu::gen_tile_full_prior_covariance,
                               gen_tile_full_prior_covariance_distributed_action,
                               "gen_tile_full_prior_covariance");
GPRAT_DEFINE_PLAIN_ACTION_FOR(&cpu::gen_tile_prior_covariance);

hpx::future<tile_handle<double>> gen_tile_full_prior_covariance_distributed(
    const tile_handle<double> &tile,
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const SEKParams &sek_params,
    const std::vector<double> &input)
{
    return tile.set_async(cpu::gen_tile_full_prior_covariance(row, col, N, n_regressors, sek_params, input));
}

hpx::future<tile_handle<double>> gen_tile_cross_covariance_distributed(
    const tile_handle<double> &tile,
    std::size_t row,
    std::size_t col,
    std::size_t N_row,
    std::size_t N_col,
    std::size_t n_regressors,
    const SEKParams &sek_params,
    const std::vector<double> &row_input,
    const std::vector<double> &col_input);

HPX_DEFINE_PLAIN_DIRECT_ACTION(gen_tile_cross_covariance_distributed);
GPRAT_DECLARE_PLAIN_ACTION_FOR(&cpu::gen_tile_cross_covariance,
                               gen_tile_cross_covariance_distributed_action,
                               "gen_tile_cross_covariance");
GPRAT_DEFINE_PLAIN_ACTION_FOR(&cpu::gen_tile_cross_covariance);

hpx::future<tile_handle<double>> gen_tile_cross_covariance_distributed(
    const tile_handle<double> &tile,
    std::size_t row,
    std::size_t col,
    std::size_t N_row,
    std::size_t N_col,
    std::size_t n_regressors,
    const SEKParams &sek_params,
    const std::vector<double> &row_input,
    const std::vector<double> &col_input)
{
    return tile.set_async(
        cpu::gen_tile_cross_covariance(row, col, N_row, N_col, n_regressors, sek_params, row_input, col_input));
}

hpx::future<tile_handle<double>> gen_tile_transpose_distributed(
    const tile_handle<double> &tile, std::size_t N_row, std::size_t N_col, const tile_handle<double> &src);
HPX_DEFINE_PLAIN_DIRECT_ACTION(gen_tile_transpose_distributed);
GPRAT_DECLARE_PLAIN_ACTION_FOR(&cpu::gen_tile_transpose, gen_tile_transpose_distributed_action, "gen_tile_transpose");
GPRAT_DEFINE_PLAIN_ACTION_FOR(&cpu::gen_tile_transpose);

hpx::future<tile_handle<double>> gen_tile_transpose_distributed(
    const tile_handle<double> &tile, std::size_t N_row, std::size_t N_col, const tile_handle<double> &src)
{
    return hpx::dataflow(
        hpx::launch::async,
        [=](hpx::future<mutable_tile_data<double>> &&tiled)
        { return tile.set_async(cpu::gen_tile_transpose(N_row, N_col, tiled.get())); },
        src.get_async());
}

hpx::future<tile_handle<double>> gen_tile_output_distributed(
    const tile_handle<double> &tile, std::size_t row, std::size_t N, const std::vector<double> &output);

HPX_DEFINE_PLAIN_DIRECT_ACTION(gen_tile_output_distributed);
GPRAT_DECLARE_PLAIN_ACTION_FOR(&cpu::gen_tile_output, gen_tile_output_distributed_action, "gen_tile_output");
GPRAT_DEFINE_PLAIN_ACTION_FOR(&cpu::gen_tile_output);

hpx::future<tile_handle<double>> gen_tile_output_distributed(
    const tile_handle<double> &tile, std::size_t row, std::size_t N, const std::vector<double> &output)
{
    return tile.set_async(cpu::gen_tile_output(row, N, output));
}

hpx::future<tile_handle<double>> gen_tile_grad_l_distributed(
    const tile_handle<double> &tile,
    std::size_t N,
    const SEKParams &sek_params,
    const const_tile_data<double> &distance);
HPX_DEFINE_PLAIN_DIRECT_ACTION(gen_tile_grad_l_distributed);
GPRAT_DECLARE_PLAIN_ACTION_FOR(&cpu::gen_tile_grad_l, gen_tile_grad_l_distributed_action, "gen_tile_grad_l");
GPRAT_DEFINE_PLAIN_ACTION_FOR(&cpu::gen_tile_grad_l);

hpx::future<tile_handle<double>> gen_tile_grad_l_distributed(
    const tile_handle<double> &tile,
    std::size_t N,
    const SEKParams &sek_params,
    const const_tile_data<double> &distance)
{
    return tile.set_async(cpu::gen_tile_grad_l(N, sek_params, distance));
}

hpx::future<tile_handle<double>> gen_tile_grad_v_distributed(
    const tile_handle<double> &tile,
    std::size_t N,
    const SEKParams &sek_params,
    const const_tile_data<double> &distance);
HPX_DEFINE_PLAIN_DIRECT_ACTION(gen_tile_grad_v_distributed);
GPRAT_DECLARE_PLAIN_ACTION_FOR(&cpu::gen_tile_grad_v, gen_tile_grad_v_distributed_action, "gen_tile_grad_v");
GPRAT_DEFINE_PLAIN_ACTION_FOR(&cpu::gen_tile_grad_l);

hpx::future<tile_handle<double>> gen_tile_grad_v_distributed(
    const tile_handle<double> &tile,
    std::size_t N,
    const SEKParams &sek_params,
    const const_tile_data<double> &distance)
{
    return tile.set_async(cpu::gen_tile_grad_v(N, sek_params, distance));
}

hpx::future<tile_handle<double>> gen_tile_zeros_distributed(const tile_handle<double> &tile, std::size_t N);

HPX_DEFINE_PLAIN_DIRECT_ACTION(gen_tile_zeros_distributed);
GPRAT_DECLARE_PLAIN_ACTION_FOR(&cpu::gen_tile_zeros, gen_tile_zeros_distributed_action, "gen_tile_output");
GPRAT_DEFINE_PLAIN_ACTION_FOR(&cpu::gen_tile_zeros);

hpx::future<tile_handle<double>> gen_tile_zeros_distributed(const tile_handle<double> &tile, std::size_t N)
{
    return tile.set_async(cpu::gen_tile_zeros(N));
}

hpx::future<tile_handle<double>> gen_tile_identity_distributed(const tile_handle<double> &tile, std::size_t N);

HPX_DEFINE_PLAIN_DIRECT_ACTION(gen_tile_identity_distributed);
GPRAT_DECLARE_PLAIN_ACTION_FOR(&cpu::gen_tile_identity, gen_tile_identity_distributed_action, "gen_tile_identity");
GPRAT_DEFINE_PLAIN_ACTION_FOR(&cpu::gen_tile_identity);

hpx::future<tile_handle<double>> gen_tile_identity_distributed(const tile_handle<double> &tile, std::size_t N)
{
    return tile.set_async(cpu::gen_tile_identity(N));
}

hpx::future<tile_handle<double>> get_matrix_diagonal_distributed(const tile_handle<double> &A, std::size_t M);

HPX_DEFINE_PLAIN_DIRECT_ACTION(get_matrix_diagonal_distributed);
GPRAT_DECLARE_PLAIN_ACTION_FOR(&cpu::get_matrix_diagonal,
                               get_matrix_diagonal_distributed_action,
                               "get_matrix_diagonal");
GPRAT_DEFINE_PLAIN_ACTION_FOR(&cpu::get_matrix_diagonal);

hpx::future<tile_handle<double>> get_matrix_diagonal_distributed(const tile_handle<double> &A, std::size_t M)
{
    return hpx::dataflow(
        hpx::launch::async,
        [A, M](hpx::future<mutable_tile_data<double>> &&Ad)
        { return A.set_async(cpu::get_matrix_diagonal(Ad.get(), M)); },
        A.get_async());
}

hpx::future<double> compute_loss_distributed(const tile_handle<double> &K_diag_tile,
                                             const tile_handle<double> &alpha_tile,
                                             const tile_handle<double> &y_tile,
                                             std::size_t N);

HPX_DEFINE_PLAIN_DIRECT_ACTION(compute_loss_distributed);
GPRAT_DECLARE_PLAIN_ACTION_FOR(&cpu::compute_loss, compute_loss_distributed_action, "compute_loss");
GPRAT_DEFINE_PLAIN_ACTION_FOR(&cpu::get_matrix_diagonal);

hpx::future<double> compute_loss_distributed(const tile_handle<double> &K_diag_tile,
                                             const tile_handle<double> &alpha_tile,
                                             const tile_handle<double> &y_tile,
                                             std::size_t N)
{
    return hpx::dataflow(
        hpx::launch::async,
        [=](hpx::future<mutable_tile_data<double>> &&K_diag_tiled,
            hpx::future<mutable_tile_data<double>> &&alpha_tiled,
            hpx::future<mutable_tile_data<double>> &&y_tiled)
        { return cpu::compute_loss(K_diag_tiled.get(), alpha_tiled.get(), y_tiled.get(), N); },
        K_diag_tile.get_async(),
        alpha_tile.get_async(),
        y_tile.get_async());
}

hpx::future<double> compute_trace_distributed(const tile_handle<double> &diagonal, double trace);

HPX_DEFINE_PLAIN_DIRECT_ACTION(compute_trace_distributed);
GPRAT_DECLARE_PLAIN_ACTION_FOR(&cpu::compute_trace, compute_trace_distributed_action, "compute_loss");
GPRAT_DEFINE_PLAIN_ACTION_FOR(&cpu::compute_trace);

hpx::future<double> compute_trace_distributed(const tile_handle<double> &diagonal, double trace)
{
    return hpx::dataflow(
        hpx::launch::async,
        [=](hpx::future<mutable_tile_data<double>> &&diagonald) { return cpu::compute_trace(diagonald.get(), trace); },
        diagonal.get_async());
}

hpx::future<double>
compute_dot_distributed(const tile_handle<double> &vector_T, const tile_handle<double> &vector, double result);

HPX_DEFINE_PLAIN_DIRECT_ACTION(compute_dot_distributed);
GPRAT_DECLARE_PLAIN_ACTION_FOR(&cpu::compute_dot, compute_dot_distributed_action, "compute_loss");
GPRAT_DEFINE_PLAIN_ACTION_FOR(&cpu::compute_dot);

hpx::future<double>
compute_dot_distributed(const tile_handle<double> &vector_T, const tile_handle<double> &vector, double result)
{
    return hpx::dataflow(
        hpx::launch::async,
        [=](hpx::future<mutable_tile_data<double>> &&vector_Td, hpx::future<mutable_tile_data<double>> &&vectord)
        { return cpu::compute_dot(vector_Td.get(), vectord.get(), result); },
        vector_T.get_async(),
        vector.get_async());
}

hpx::future<double> compute_trace_diag_distributed(const tile_handle<double> &tile, double trace, std::size_t N);

HPX_DEFINE_PLAIN_DIRECT_ACTION(compute_trace_diag_distributed);
GPRAT_DECLARE_PLAIN_ACTION_FOR(&cpu::compute_trace_diag, compute_trace_diag_distributed_action, "compute_loss");
GPRAT_DEFINE_PLAIN_ACTION_FOR(&cpu::compute_trace_diag);

hpx::future<double> compute_trace_diag_distributed(const tile_handle<double> &tile, double trace, std::size_t N)
{
    return hpx::dataflow(
        hpx::launch::async,
        [=](hpx::future<mutable_tile_data<double>> &&tiled) { return cpu::compute_trace_diag(tiled.get(), trace, N); },
        tile.get_async());
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

void finish_step(const char *name, double elapsed_seconds)
{
    std::cerr << name << " done in " << elapsed_seconds << " seconds" << std::endl;
    hpx::evaluate_active_counters(true, name);
}

void run(hpx::program_options::variables_map &vm)
{
    /////////////////////
    /////// configuration
    const std::size_t START = vm["start"].as<std::size_t>();
    const std::size_t END = vm["end"].as<std::size_t>();
    const std::size_t STEP = vm["step"].as<std::size_t>();
    const std::size_t LOOP = vm["loop"].as<std::size_t>();
    const std::size_t OPT_ITER = vm["opt_iter"].as<std::size_t>();
    const std::size_t enabled = vm["enabled"].as<std::size_t>();

    const std::size_t n_test = vm["n_test"].as<std::size_t>();
    const std::size_t n_tiles = vm["tiles"].as<std::size_t>();
    const std::size_t n_reg = vm["regressors"].as<std::size_t>();

    const auto &train_path = vm["train_x_path"].as<std::string>();
    const auto &out_path = vm["train_y_path"].as<std::string>();
    const auto &test_path = vm["test_path"].as<std::string>();

    std::optional<gprat_results> test_results;
    const auto test_results_path = vm["test_results_path"].as<std::string>();
    if (!test_results_path.empty())
    {
        test_results = load_test_data_results(test_results_path);
        std::cerr << "We have comparison data!" << std::endl;
    }

    tiled_scheduler_sma scheduler;

    for (std::size_t start = START; start <= END; start = start * STEP)
    {
        const auto n_train = start;
        for (std::size_t l = 0; l < LOOP; l++)
        {
            hpx::chrono::high_resolution_timer total_timer;

            // Compute tile sizes and number of predict tiles
            const auto tile_size = compute_train_tile_size(n_train, n_tiles);
            const auto result = compute_test_tiles(n_test, n_tiles, tile_size);
            /////////////////////
            ///// hyperparams
            AdamParams hpar = { 0.1, 0.9, 0.999, 1e-8, OPT_ITER };
            SEKParams sek_params = { 1.0, 1.0, 0.1 };
            std::vector<bool> trainable = { true, true, true };

            /////////////////////
            ////// data loading
            GP_data training_input(train_path, n_train, n_reg);
            GP_data training_output(out_path, n_train, n_reg);
            GP_data test_input(test_path, n_test, n_reg);

            /////////////////////
            ///// GP
            gprat_results results;

            // Start with a clean slate
            hpx::reset_active_counters();

            hpx::chrono::high_resolution_timer cholesky_timer;
            if (enabled & (1 << 0))
            {
                results.choleksy =
                    to_vector(cpu::cholesky(scheduler, training_input.data, sek_params, n_tiles, tile_size, n_reg));
            }
            const auto cholesky_time = cholesky_timer.elapsed();
            finish_step("cholesky", cholesky_time);

            hpx::chrono::high_resolution_timer opt_timer;
            if (enabled & (1 << 1))
            {
                results.losses = cpu::optimize(
                    scheduler,
                    training_input.data,
                    training_output.data,
                    n_tiles,
                    tile_size,
                    n_reg,
                    hpar,
                    sek_params,
                    trainable);
            }
            const auto opt_time = opt_timer.elapsed();
            finish_step("opt", opt_time);

            hpx::chrono::high_resolution_timer predict_timer;
            if (enabled & (1 << 2))
            {
                results.pred = cpu::predict(
                    scheduler,
                    training_input.data,
                    training_output.data,
                    test_input.data,
                    sek_params,
                    n_tiles,
                    tile_size,
                    result.first,
                    result.second,
                    n_reg);
            }
            const auto predict_time = predict_timer.elapsed();
            finish_step("predict", predict_time);

            hpx::chrono::high_resolution_timer predict_with_uncertainty_timer;
            if (enabled & (1 << 3))
            {
                results.sum = cpu::predict_with_uncertainty(
                    scheduler,
                    training_input.data,
                    training_output.data,
                    test_input.data,
                    sek_params,
                    n_tiles,
                    tile_size,
                    result.first,
                    result.second,
                    n_reg);
            }
            const auto predict_with_uncertainty_time = predict_with_uncertainty_timer.elapsed();
            finish_step("predict_with_uncertainty", predict_with_uncertainty_time);

            hpx::chrono::high_resolution_timer predict_with_full_cov_timer;
            if (enabled & (1 << 4))
            {
                results.full = cpu::predict_with_full_cov(
                    scheduler,
                    training_input.data,
                    training_output.data,
                    test_input.data,
                    sek_params,
                    n_tiles,
                    tile_size,
                    result.first,
                    result.second,
                    n_reg);
            }
            const auto predict_with_full_cov_time = predict_with_full_cov_timer.elapsed();
            finish_step("predict_with_full_cov", predict_with_full_cov_time);

            // Save parameters and times to a .csv file with a header
            std::ofstream outfile(vm["timings_csv"].as<std::string>(), std::ios::app);
            if (outfile.tellp() == 0)
            {
                // If file is empty, write the header
                outfile << "Cores,N_train,N_test,N_tiles,N_regressor,Opt_iter,Total_time,Init_time,Cholesky_time,"
                           "Opt_time,Pred_Uncer_time,Pred_Full_time,Pred_time,N_loop\n";
            }
            outfile << hpx::get_locality_id() << "," << n_train << "," << n_test << "," << n_tiles << "," << n_reg
                    << "," << OPT_ITER << "," << total_timer.elapsed() << "," << 0 << "," << cholesky_time << ","
                    << opt_time << "," << predict_with_uncertainty_time << "," << predict_with_full_cov_time << ","
                    << predict_time << "," << l << "\n";
            outfile.close();

            if (test_results)
            {
#define REQUIRE(expr)                                                                                                  \
    if (!expr)                                                                                                         \
        throw std::runtime_error(#expr);
#define REQUIRE_THAT(a, b)                                                                                             \
    if (!b.match(a))                                                                                                   \
        throw std::runtime_error(std::format("{} != {}: {} {}", #a, #b, a, b.describe()));
                const auto &expected_results = *test_results;
                std::cerr << "Validating results..." << std::endl;
                REQUIRE(results.choleksy.size() == expected_results.choleksy.size());
                REQUIRE(results.losses.size() == expected_results.losses.size());
                REQUIRE(results.sum.size() == expected_results.sum.size());
                REQUIRE(results.sum[0].size() == expected_results.sum[0].size());
                REQUIRE(results.full.size() == expected_results.full.size());
                REQUIRE(results.full[0].size() == expected_results.full[0].size());
                REQUIRE(results.pred.size() == expected_results.pred.size());

                // Now we can compare content
                // The default-constructed WithinRel() matcher has a tolerance of epsilon * 100
                // see:
                // https://github.com/catchorg/Catch2/blob/914aeecfe23b1e16af6ea675a4fb5dbd5a5b8d0a/docs/comparing-floating-point-numbers.md#withinrel
                using Catch::Matchers::WithinRel;
                double eps = std::numeric_limits<double>::epsilon() * 1'000'000;
                for (std::size_t i = 0, n = results.choleksy.size(); i != n; ++i)
                {
                    for (std::size_t j = 0, m = results.choleksy[i].size(); j != m; ++j)
                    {
                        REQUIRE_THAT(results.choleksy[i][j], WithinRel(expected_results.choleksy[i][j], eps));
                    }
                }
                for (std::size_t i = 0, n = results.losses.size(); i != n; ++i)
                {
                    REQUIRE_THAT(results.losses[i], WithinRel(expected_results.losses[i], eps));
                }

                for (std::size_t i = 0, n = results.full.size(); i != n; ++i)
                {
                    for (std::size_t j = 0, m = results.full[i].size(); j != m; ++j)
                    {
                        REQUIRE_THAT(results.full[i][j], WithinRel(expected_results.full[i][j], eps));
                    }
                }

                for (std::size_t i = 0, n = results.sum.size(); i != n; ++i)
                {
                    for (std::size_t j = 0, m = results.sum[i].size(); j != m; ++j)
                    {
                        REQUIRE_THAT(results.sum[i][j], WithinRel(expected_results.sum[i][j], eps));
                    }
                }

                for (std::size_t i = 0, n = results.pred.size(); i != n; ++i)
                {
                    REQUIRE_THAT(results.pred[i], WithinRel(expected_results.pred[i], eps));
                }
            }

            std::cerr << "====================" << std::endl;
        }
    }
    std::cerr << "DONE!" << std::endl;
}

void startup()
{
    std::cerr << "startup() called" << std::endl;

    static struct once_dummy_struct
    {
        once_dummy_struct()
        {
            register_performance_counters();
            register_distributed_tile_counters();
        }
    } once_dummy;
}

bool check_startup(hpx::startup_function_type &startup_func, bool &pre_startup)
{
    // perform full module startup (counters will be used)
    startup_func = startup;
    pre_startup = true;
    return true;
}

GPRAT_NS_END

HPX_REGISTER_ACTION(GPRAT_NS::gen_tile_covariance_distributed_action);

HPX_REGISTER_STARTUP_MODULE(GPRAT_NS::check_startup)

int hpx_main(hpx::program_options::variables_map &vm)
{
    hpx::get_runtime().get_config().dump(0, std::cerr);
    std::cerr << "OS Threads: " << hpx::get_os_thread_count() << std::endl;
    std::cerr << "All localities: " << hpx::get_num_localities().get() << std::endl;
    std::cerr << "Root locality: " << hpx::find_root_locality() << std::endl;
    std::cerr << "This locality: " << hpx::find_here() << std::endl;
    std::cerr << "Remote localities: " << hpx::find_remote_localities().size() << std::endl;

    auto numa_domains = hpx::compute::host::numa_domains();
    std::cerr << "Local NUMA domains: " << numa_domains.size() << std::endl;
    for (const auto &domain : numa_domains)
    {
        const auto &num_pus = domain.num_pus();
        std::cerr << " Domain: " << num_pus.first << " " << num_pus.second << std::endl;
    }

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
        ("n_test", po::value<std::size_t>()->default_value(128), "Number of test samples")
        ("loop", po::value<std::size_t>()->default_value(1), "Number of iterations to be performed for each number of training samples")
        ("opt_iter", po::value<std::size_t>()->default_value(3), "Number of optimization iterations*/")
        ("enabled", po::value<std::size_t>()->default_value((std::numeric_limits<std::size_t>::max)()), "Bitmask of enabled steps")
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
