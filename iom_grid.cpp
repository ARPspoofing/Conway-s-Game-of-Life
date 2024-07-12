#include "iom.h"

const int DIRECTIONS[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
const int VALID_LOOKUP[2] = {0, 1};
const int GRANULARITY = 2;

int iom(int nThreads, int nGenerations, std::vector<std::vector<int>> &startWorld, int nRows, int nCols, int nInvasions, std::vector<int> invasionTimes, std::vector<std::vector<std::vector<int>>> invasionPlans) {
    std::vector<std::vector<int>> world1 = startWorld;
    std::vector<std::vector<int>> world2(nRows, std::vector<int>(nCols));
    // Cells that died by fighting, cleared at the end of every iteration
    std::set<std::pair<int, int>> cells_died_by_fighting;
    std::vector<std::vector<int>> *current_generation = &world1;
    std::vector<std::vector<int>> *next_generation = &world2;
    std::vector<Grid> grids = divideMatrix(nRows, nCols, nThreads, GRANULARITY);  // adjust granularity as needed

    int invasion_pointer = 0;
    int total_death_count = 0;

    for (int g = 1; g <= nGenerations; g++) {

        int generational_death_count = 0;
        thread_local std::unordered_map<int, int> neighbour_counts;

        generational_death_count = updateNextGeneration(*current_generation, *next_generation, cells_died_by_fighting, grids, nRows, nCols, nThreads);
        total_death_count += generational_death_count;

        // Check if it's time for an invasion
        if (invasion_pointer < nInvasions && g == invasionTimes[invasion_pointer]) {
            // Apply the invasion plan
            std::vector<std::vector<int>> &invasionPlan = invasionPlans[invasion_pointer];
            int invasion_death_count = applyInvasion(*current_generation, *next_generation, invasionPlan, cells_died_by_fighting, grids, nThreads);
            total_death_count += invasion_death_count;
            // Move to the next invasion time
            invasion_pointer++;
        }

        // Clear cells that died by fighting
        cells_died_by_fighting.clear();

        // Swap current and next generation
        std::swap(current_generation, next_generation);
    }

    printf("Death count: %d \n", total_death_count);

    //Clear memory
    world1.clear();
    world2.clear();
    cells_died_by_fighting.clear();
    invasionTimes.clear();
    invasionPlans.clear();

    return total_death_count;
}

/**
 * Updates matrix of next generation, and return total number of deaths for the generation.
 */
int updateNextGeneration(std::vector<std::vector<int>> &current_generation, std::vector<std::vector<int>> &next_generation, std::set<std::pair<int, int>> &cells_died_by_fighting, std::vector<Grid> grids, int nRows, int nCols, int nThreads) {
    int generational_death_count = 0;

    #pragma omp parallel num_threads(nThreads)
    {
        #pragma omp single
        {
            for (const Grid &g: grids) {
                #pragma omp task shared(current_generation, next_generation, cells_died_by_fighting, nRows, nCols)
                {
                    int local_death_count = 0; // Local death count

                    for (int i = g.startRow; i < g.endRow; i++) {
                        for (int j = g.startCol; j < g.endCol; j++) {
                            std::unordered_map<int, int> neighbour_counts = calculateNeighbourCounts(current_generation, nRows, nCols, i, j);

                            std::pair<int, int> valid = checkValidity(neighbour_counts, cells_died_by_fighting, current_generation[i][j], i, j);

                            next_generation[i][j] = valid.first;

                            local_death_count += VALID_LOOKUP[valid.second];
                        }
                    }
                    #pragma omp atomic
                    generational_death_count += local_death_count;
                }
            }
        }
    }

    return generational_death_count;
}


/**
 * Applies invasion plan for matrix.
 */
int applyInvasion(std::vector<std::vector<int>> &current_generation, std::vector<std::vector<int>> &next_generation, std::vector<std::vector<int>> &invasion_plan, std::set<std::pair<int, int>> &cells_died_by_fighting, std::vector<Grid> grids, int nThreads) {
    int invasion_death_count = 0;

    #pragma omp parallel num_threads(nThreads)
    {
        #pragma omp single
        {
            for (const Grid &g : grids) {
                #pragma omp task shared(current_generation, next_generation, invasion_plan, cells_died_by_fighting)
                {
                    int local_death_count = 0; // Local death count

                    for (int i = g.startRow; i < g.endRow; i++) {
                        for (int j = g.startCol; j < g.endCol; j++) {
                            
                            if (invasion_plan[i][j] == 0) {
                                continue; // Do nothing, skip this cell
                            }

                            if (current_generation[i][j] != 0 && next_generation[i][j] != 0) {
                                // Was originally occupied by some faction, so reproduction does not apply.
                                // Only survival can lead to next_generation cell being not 0.
                                local_death_count += 1;
                            } else if (
                                current_generation[i][j] != 0 && cells_died_by_fighting.find({i, j}) == cells_died_by_fighting.end()) {
                                // Cell was originally occupied, but DID NOT die by in-fighting
                                local_death_count += 1;
                            }

                            // Apply invasion
                            next_generation[i][j] = invasion_plan[i][j];
                        }
                    }
                    #pragma omp atomic
                    invasion_death_count += local_death_count;
                }
            }
        }
    }

    return invasion_death_count;
}

inline std::unordered_map<int, int> calculateNeighbourCounts(const std::vector<std::vector<int>> &world, int nRows, int nCols, int i, int j) {
    std::unordered_map<int, int> neighbour_counts;
    int curr_faction = world[i][j];

    for (int k = 0; k < 8; k++) {
        int new_i = (i + DIRECTIONS[k][0] + nRows) % nRows;
        int new_j = (j + DIRECTIONS[k][1] + nCols) % nCols;
        int neighbour = world[new_i][new_j];
        neighbour_counts[neighbour]++;

        //Stop checking when there is at least a hostile neighbour for a live cell, 
        if (curr_faction != neighbour && neighbour != 0 && curr_faction != 0) {
            break;
        }
    }

    return neighbour_counts;
}


std::pair<int, int> checkValidity(std::unordered_map<int, int> &neighbour_counts, std::set<std::pair<int, int>> &cells_died_by_fighting, int cell_alive, int i, int j) {
    // First item: New cell value, Second item: Whether the cell died due to infighting
    std::pair<int, int> result = {0, 0};

    // Reproduction Case only happens if cell was not alive before
    if (cell_alive == 0) {
        int higher_faction = 0; // Initialize to a default value of 0
        // Iterate through the unordered_map and check for the higher faction with a value of 3
        for (const auto &pair : neighbour_counts) {
            if (pair.first != 0 && pair.second == 3) {
                higher_faction = std::max(higher_faction, pair.first);
            }
        }
        result.first = higher_faction;
        return result;
    }

    int friendly_neighbours = 0;
    int hostile_neighbours = 0;

    for (const auto &pair : neighbour_counts) {
        if (pair.first == cell_alive) {
            friendly_neighbours += pair.second;
        } else if (pair.first != 0) {
            // Don't need to consider pair.first == 0
            hostile_neighbours += pair.second;
            // Stop checking the moment there is a hostile neighbour
            break;
        }
    }

    // Check for infighting (at least 1 hostile neighbor)
    if (hostile_neighbours >= 1) {
        result.second = true;
        #pragma omp critical
        cells_died_by_fighting.insert({i, j});
        return result;
    }

    // Check for survival (no hostile neighbors)
    if (friendly_neighbours == 2 || friendly_neighbours == 3) {
        result.first = cell_alive;
    }

    return result;
}

std::vector<Grid> divideMatrix(int nRows, int nCols, int nThreads, int granularity) {
    std::vector<Grid> grids;
    int totalTasks = nThreads * granularity;

    // Calculate the average number of cells in each grid
    int avgCellsPerGrid = (nRows * nCols) / totalTasks;

    // Estimate the dimensions for each grid. 
    // This is a basic estimation. Depending on the matrix size and totalTasks, 
    // adjustments might be needed to ensure all cells are covered without overlaps.
    int gridRows = sqrt(avgCellsPerGrid);
    int gridCols = avgCellsPerGrid / gridRows;

    // Using these estimated dimensions, create the grids
    for (int r = 0; r < nRows; r += gridRows) {
        for (int c = 0; c < nCols; c += gridCols) {
            Grid g;
            g.startRow = r;
            g.endRow = std::min(r + gridRows, nRows);
            g.startCol = c;
            g.endCol = std::min(c + gridCols, nCols);
            grids.push_back(g);
        }
    }

    return grids;
}
