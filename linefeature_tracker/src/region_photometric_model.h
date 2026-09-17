#pragma once

#include <algorithm>
#include <array>
#include <cmath>

struct RegionPhotometricModel
{
    static constexpr int kRows = 3;
    static constexpr int kCols = 4;
    static constexpr int kRegionCount = kRows * kCols;

    int image_width;
    int image_height;

    std::array<double, kRegionCount> gain;
    std::array<double, kRegionCount> bias;
    std::array<int, kRegionCount> sample_count;
    std::array<int, kRegionCount> valid;

    RegionPhotometricModel()
    {
        reset(0, 0);
    }

    void reset(int width, int height)
    {
        image_width = width;
        image_height = height;

        gain.fill(1.0);
        bias.fill(0.0);
        sample_count.fill(0);
        valid.fill(0);
    }

    int index(int row, int col) const
    {
        row = std::max(0, std::min(kRows - 1, row));
        col = std::max(0, std::min(kCols - 1, col));

        return row * kCols + col;
    }

    int regionIndex(float x, float y) const
    {
        if (image_width <= 0 || image_height <= 0)
            return 0;

        int col = static_cast<int>(
            x * static_cast<float>(kCols) / image_width);

        int row = static_cast<int>(
            y * static_cast<float>(kRows) / image_height);

        return index(row, col);
    }

    double gainAt(float x, float y) const
    {
        return interpolate(gain, x, y);
    }

    double biasAt(float x, float y) const
    {
        return interpolate(bias, x, y);
    }

private:
    double interpolate(
        const std::array<double, kRegionCount> &values,
        float x,
        float y) const
    {
        if (image_width <= 0 || image_height <= 0)
            return values[0];

        // 将像素位置转换到区域中心坐标。
        const double grid_x =
            static_cast<double>(x) * kCols / image_width - 0.5;

        const double grid_y =
            static_cast<double>(y) * kRows / image_height - 0.5;

        int col0 = static_cast<int>(std::floor(grid_x));
        int row0 = static_cast<int>(std::floor(grid_y));

        int col1 = col0 + 1;
        int row1 = row0 + 1;

        double tx = grid_x - col0;
        double ty = grid_y - row0;

        if (col0 < 0)
        {
            col0 = 0;
            col1 = 0;
            tx = 0.0;
        }
        else if (col0 >= kCols - 1)
        {
            col0 = kCols - 1;
            col1 = col0;
            tx = 0.0;
        }

        if (row0 < 0)
        {
            row0 = 0;
            row1 = 0;
            ty = 0.0;
        }
        else if (row0 >= kRows - 1)
        {
            row0 = kRows - 1;
            row1 = row0;
            ty = 0.0;
        }

        const double value00 = values[index(row0, col0)];
        const double value01 = values[index(row0, col1)];
        const double value10 = values[index(row1, col0)];
        const double value11 = values[index(row1, col1)];

        const double value_top =
            (1.0 - tx) * value00 + tx * value01;

        const double value_bottom =
            (1.0 - tx) * value10 + tx * value11;

        return (1.0 - ty) * value_top + ty * value_bottom;
    }
};
