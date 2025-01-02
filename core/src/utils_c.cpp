#include "../include/utils_c.hpp"

#include <cstdio>

namespace utils
{

/**
 * @brief Compute the number of tiles for training data, given the number of
 * samples and the size of each tile.
 *
 * @param n_samples Number of samples
 * @param n_tile_size Size of each tile
 */
int compute_train_tiles(int n_samples, int n_tile_size)
{
    if (n_tile_size > 0)
    {
        std::size_t _n_samples = static_cast<std::size_t>(n_samples);
        std::size_t _n_tile_size = static_cast<std::size_t>(n_tile_size);
        std::size_t _n_tiles = _n_samples / _n_tile_size;
        return static_cast<int>(_n_tiles);
    }
    else
    {
        throw std::runtime_error("Error: Please specify a valid value for train_tile_size.\n");
    }
}

/**
 * @brief Compute the number of tiles for training data, given the number of
 * samples and the size of each tile.
 *
 * @param n_samples Number of samples
 * @param n_tile_size Size of each tile
 */
int compute_train_tile_size(int n_samples, int n_tiles)
{
    if (n_tiles > 0)
    {
        std::size_t _n_samples = static_cast<std::size_t>(n_samples);
        std::size_t _n_tiles = static_cast<std::size_t>(n_tiles);
        std::size_t _n_tile_size = _n_samples / n_tiles;
        return static_cast<int>(_n_tile_size);
    }
    else
    {
        throw std::runtime_error("Error: Please specify a valid value for train_tiles.\n");
    }
}

/**
 * @brief Compute the number of test tiles and the size of a test tile.
 *
 * Uses n_tiles_size if n_test is divisible by n_tile_size. Otherwise uses
 * n_tiles for calculation.
 *
 * @param n_test Number of test samples
 * @param n_tiles Number of tiles
 * @param n_tile_size Size of each tile
 */
std::pair<int, int> compute_test_tiles(int n_test, int n_tiles, int n_tile_size)
{
    std::size_t _n_test = static_cast<std::size_t>(n_test);
    std::size_t _n_tiles = static_cast<std::size_t>(n_tiles);
    std::size_t _n_tile_size = static_cast<std::size_t>(n_tile_size);
    std::size_t m_tiles;
    std::size_t m_tile_size;

    if ((_n_test % _n_tile_size) > 0)
    {
        m_tiles = _n_tiles;
        m_tile_size = _n_test / m_tiles;
    }
    else
    {
        m_tiles = _n_test / _n_tile_size;
        m_tile_size = _n_tile_size;
    }
    return { static_cast<int>(m_tiles), static_cast<int>(m_tile_size) };
}

/**
 * @brief Load data from file
 *
 * @param file_path Path to the file
 * @param n_samples Number of samples to load
 */
std::vector<double> load_data(const std::string &file_path, int n_samples, int offset)
{
    std::vector<double> _data;
    _data.resize(n_samples + offset, 0.0);

    FILE *input_file = fopen(file_path.c_str(), "r");
    if (input_file == NULL)
    {
        throw std::runtime_error("Error: File not found: " + file_path);
    }

    // load data
    std::size_t scanned_elements = 0;
    for (int i = 0; i < n_samples; i++)
    {
        scanned_elements += fscanf(input_file, "%lf", &_data[i+offset]);  // scanned_elements++;
    }

    fclose(input_file);

    if (scanned_elements != n_samples)
    {
        throw std::runtime_error("Error: Data not correctly read. Expected "
                                 + std::to_string(n_samples)
                                 + " elements, but read "
                                 + std::to_string(scanned_elements));
    }

    return std::move(_data);
}

/**
 * @brief Print a vector
 *
 * @param vec Vector to print
 * @param start Start index
 * @param end End index
 * @param separator Separator between elements
 */
void print(const std::vector<double> &vec, int start, int end, const std::string &separator)
{
    // Convert negative indices to positive
    if (start < 0)
    {
        start += vec.size();
    }
    if (end < 0)
    {
        end += vec.size() + 1;
    }

    // Ensure the indices are within bounds
    if (start < 0)
    {
        start = 0;
    }
    if (end > vec.size())
    {
        end = vec.size();
    }

    // Validate the range
    if (start >= vec.size() || start >= end)
    {
        std::cerr << "Invalid range" << std::endl;
        return;
    }

    for (int i = start; i < end; i++)
    {
        std::cout << vec[i];
        if (i < end - 1)
        {
            std::cout << separator;
        }
    }
    std::cout << std::endl;
}

// Start HPX runtime
void start_hpx_runtime(int argc, char **argv)
{
    hpx::start(nullptr, argc, argv);
}

// Resume HPX runtime
void resume_hpx_runtime()
{
    hpx::resume();
}

// Wait for all tasks to finish, and suspend the HPX runtime
void suspend_hpx_runtime()
{
    hpx::suspend();
}

// Stop HPX runtime
void stop_hpx_runtime()
{
    hpx::post([]()
              { hpx::finalize(); });
    hpx::stop();
}
}  // namespace utils
