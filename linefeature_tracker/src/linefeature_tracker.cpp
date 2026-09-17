#include "linefeature_tracker.h"
#include <math.h>
#include <fstream>
#include <mutex>
// #include "line_descriptor/src/precomp_custom.hpp"

vector<vector<double>> param;
namespace
{
std::mutex g_line_diag_mutex;
std::ofstream g_line_diag_file;
long long g_line_diag_frame = -1;
}

LineFeatureTracker::LineFeatureTracker()
{
    allfeature_cnt = 0;
    frame_cnt = 0;
    sum_time = 0.0;
}

void LineFeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());

    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
    K_ = m_camera->initUndistortRectifyMap(undist_map1_, undist_map2_);
}

vector<Line> &LineFeatureTracker::normalizedLineEndPoints()
{
    float fx = K_.at<float>(0, 0);
    float fy = K_.at<float>(1, 1);
    float cx = K_.at<float>(0, 2);
    float cy = K_.at<float>(1, 2);
    for (unsigned int i = 0; i < cur_img->vecLine.size(); i++)
    {
        cur_img->vecLine[i].norm_start.x = (cur_img->vecLine[i].StartPt.x - cx) / fx;
        cur_img->vecLine[i].norm_start.y = (cur_img->vecLine[i].StartPt.y - cy) / fy;
        cur_img->vecLine[i].norm_end.x = (cur_img->vecLine[i].EndPt.x - cx) / fx;
        cur_img->vecLine[i].norm_end.y = (cur_img->vecLine[i].EndPt.y - cy) / fy;
    }
    return cur_img->vecLine;
}
#if _SHITONG_

void LineFeatureTracker::setMask()
{

    mask = cv::Mat(ROW, COL, CV_8UC1, 255);

    for (unsigned int i = 0; i < forw_img->vecLine.size(); i++)
    {
        cv::Point start = cv::Point(forw_img->vecLine[i].keyPoint[1].x, forw_img->vecLine[i].keyPoint[1].y);
        cv::Point end = cv::Point(forw_img->vecLine[i].keyPoint[forw_img->vecLine[i].keyPoint.size() - 1].x, forw_img->vecLine[i].keyPoint[forw_img->vecLine[i].keyPoint.size() - 1].y);

        cv::line(mask, start, end, 0, 40);
    }

    // imshow("mask", mask);
    // waitKey();
}

void LineFeatureTracker::setMerge(cv::Mat &mergel)
{

    merge = cv::Mat(ROW, COL, CV_8UC1, 255);
    int lineIndex = 0;

    for (unsigned int i = 0; i < forw_img->vecLine.size(); i++)
    {
        drawLine(forw_img->vecLine[i], merge, lineIndex);
        drawLine(forw_img->vecLine[i], mergel, lineIndex++);
    }

    // imshow("merge", merge);
    // waitKey();
}

inline float GetPixelValue(const cv::Mat &img, float x, float y)
{
    // boundary check
    if (x < 0)
        x = 0;
    if (y < 0)
        y = 0;
    if (x >= img.cols - 1)
        x = img.cols - 2;
    if (y >= img.rows - 1)
        y = img.rows - 2;

    float xx = x - floor(x);
    float yy = y - floor(y);
    int x_a1 = std::min(img.cols - 1, int(x) + 1);
    int y_a1 = std::min(img.rows - 1, int(y) + 1);

    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x) + xx * (1 - yy) * img.at<uchar>(y, x_a1) + (1 - xx) * yy * img.at<uchar>(y_a1, x) + xx * yy * img.at<uchar>(y_a1, x_a1);
}

inline double GetPhotometricallyCorrectedPixelValue(
    const cv::Mat &image,
    float x,
    float y,
    const RegionPhotometricModel *region_model,
    bool refinement_pass)
{
    const double gray =
        GetPixelValue(image, x, y);

    if (!refinement_pass || region_model == nullptr)
        return gray;

    const double effective_gain =
        region_model->effectiveGainAt(x, y);

    const double effective_bias =
        region_model->effectiveBiasAt(x, y);

    return effective_gain * gray + effective_bias;
}

void OpticalFlowSingleLevel(
    const Mat &magnitude,
    const Mat &img1,
    const Mat &img2,
    const vector<Line> &kp1,
    vector<Line> &kp2,
    vector<int> &success,
    bool inverse,
    bool has_initial,
    int layer,
    const RegionPhotometricModel *region_model,
    bool refinement_pass,
    bool record_diagnostics)
{
    kp2.resize(kp1.size());
    success.resize(kp1.size());

    // 只在最精细的第0层记录一次，避免每个金字塔层重复记录
    if (layer == 0 && record_diagnostics)
    {
        std::lock_guard<std::mutex> lock(g_line_diag_mutex);

        ++g_line_diag_frame;

        // 第一次运行时创建CSV文件并写入表头
        if (!g_line_diag_file.is_open())
        {
            g_line_diag_file.open(
                "/tmp/eplf_line_diag.csv",
                std::ios::out | std::ios::trunc);

            g_line_diag_file
                << "frame,line_index,x,y,"
                << "mean_squared_residual,"
                << "mean_patch_brightness_difference,"
                << "gray_rejected,final_success,"
                << "g1,g2,g3\n";
        }
    }

    // 原来的线光流计算
    OpticalFlowTracker tracker(
        magnitude,
        img1,
        img2,
        kp1,
        kp2,
        success,
        inverse,
has_initial,
layer,
region_model,
refinement_pass,
record_diagnostics);

    parallel_for_(
        Range(0, kp1.size()),
        std::bind(
            &OpticalFlowTracker::calculateOpticalFlow,
            &tracker,
            placeholders::_1));

    // 每帧计算结束后，将缓存的诊断结果写入磁盘
    if (layer == 0)
    {
        std::lock_guard<std::mutex> lock(g_line_diag_mutex);

        if (g_line_diag_file.is_open())
            g_line_diag_file.flush();
    }
}
#define _REGION_ 1

bool checkgoodLine(const cv::Mat &magnitude, const std::vector<cv::Point2f> &line)
{
    // int halflength = 3;
    // int bias = 10;
    int pointNum = 16;
    int pointNumber = 0;

    Point2f start = line[1];
    Point2f end = line[line.size() - 1];

    float dx0 = end.x - start.x;
    float dy0 = end.y - start.y;
    float l = sqrt(dx0 * dx0 + dy0 * dy0);

    Point2f DistPoint0[pointNum];

    for (int i = 0; i < pointNum; i++)
    {
        Point2f InsertPoint = Point2f(start.x + i * dx0 / pointNum, start.y + i * dy0 / pointNum);
        if (InsertPoint.x < 1 || InsertPoint.x > magnitude.cols - 1 ||
            InsertPoint.y < 1 || InsertPoint.y > magnitude.rows - 1)
        {
            continue;
        }
        DistPoint0[pointNumber++] = InsertPoint;
    }

    if (pointNumber == 0)
        return false;

    float mag = 0;
    float maxmag = 0;
    for (int i = 0; i < pointNumber; i++)
    {
        if (maxmag < abs(magnitude.at<float>(DistPoint0[i])))
            maxmag = abs(magnitude.at<float>(DistPoint0[i]));
        mag += abs(magnitude.at<float>(DistPoint0[i]) / pointNumber);
    }

    if (mag < 128)
        return false;
    return true;
}

static float huberloss(float a, float dealt)
{
    if (abs(a) <= dealt)
        return 0.5 * a * a;
    else
        return dealt * (abs(a) - 0.5 * dealt);
}

void OpticalFlowTracker::calculateOpticalFlow(const Range &range)
{
    // 这里设置了一些阈值，但是后来发现不太好用
    int half_patch_size = 4;
    int iterations = 5;
    int door = 40500; //最大cost
    double Graydoor = 5;

    for (size_t i = range.start; i < range.end; i++)
    {

        if (success[i] != -1)
        {

            auto kp = kp1[i].keyPoint;
            if (param[i].empty())
            {
                param[i].resize(3);
                param[i][0] = param[i][1] = param[i][2] = 0;
            }
            else if (layer == _LAYER_ - 1)
                param[i][0] = param[i][1] = param[i][2] = 0;
            double &g1 = param[i][0], &g2 = param[i][1], &g3 = param[i][2];

if (!refinement_pass)
{
    g1 *= 2;
    g2 *= 2;
}

            double cost = 0, lastCost = 0;
            int succ = i; // indicate if this point succeeded
            double errorGray = 0;
            bool gray_rejected = false;
            // Gauss-Newton iterations
            Eigen::Matrix3d H = Eigen::Matrix3d::Zero(); // hessian
            Eigen::Vector3d b = Eigen::Vector3d::Zero(); // bias
            Eigen::Vector3d J;
            Eigen::Vector2d J0; // jacobian
            for (int iter = 0; iter < iterations; iter++)
            {
                chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
                if (inverse == false)
                {
                    H = Eigen::Matrix3d::Zero();
                    b = Eigen::Vector3d::Zero();
                }
                else
                {
                    // only reset b
                    b = Eigen::Vector3d::Zero();
                }

                cost = 0;
                errorGray = 0;

                // compute cost and jacobian
                for (int m = 0; m < kp.size(); m++)
                {
                    double gray1 = 0, gray2 = 0;
                    // std::cout << kp.size();
                    double dealtx = (kp[m].x - kp[0].x);
                    double dealty = (kp[m].y - kp[0].y);

                    double dx, dy, dx0, dy0;
                    dx0 = dealty * g3;
                    dy0 = dealtx * g3;
                    dx = g1 + dx0;
                    dy = g2 - dy0;

                    for (int x = -half_patch_size; x <= half_patch_size; x++)
                        for (int y = -half_patch_size; y <= half_patch_size; y++)
                        {

                            double gray01 = GetPixelValue(img1, kp[m].x + x, kp[m].y + y);
                            double gray02 =
    GetPhotometricallyCorrectedPixelValue(
        img2,
        kp[m].x + x + dx,
        kp[m].y + y + dy,
        region_model,
        refinement_pass);
                            // double error = GetPixelValue(img1, kp[m].x + x, kp[m].y + y) -
                            //                GetPixelValue(img2, kp[m].x + x + dx, kp[m].y + y + dy);
                            double error = gray01 - gray02;
                            // if (iter == 0)
                            // {
                            gray1 += gray01;
                            gray2 += gray02;
                            // }
                            // Jacobian
                            if (inverse == false)
                            {
                                // double dx, dy;
                                // dx = g1 + (kp[m].y - kp[0].y) * g3;
                                // dy = g2 - (kp[m].x - kp[0].x) * g3;
                                J0 = -1.0 * Eigen::Vector2d(
    0.5 *
        (GetPhotometricallyCorrectedPixelValue(
             img2,
             kp[m].x + dx + x + 1,
             kp[m].y + dy + y,
             region_model,
             refinement_pass) -
         GetPhotometricallyCorrectedPixelValue(
             img2,
             kp[m].x + dx + x - 1,
             kp[m].y + dy + y,
             region_model,
             refinement_pass)),
    0.5 *
        (GetPhotometricallyCorrectedPixelValue(
             img2,
             kp[m].x + dx + x,
             kp[m].y + dy + y + 1,
             region_model,
             refinement_pass) -
         GetPhotometricallyCorrectedPixelValue(
             img2,
             kp[m].x + dx + x,
             kp[m].y + dy + y - 1,
             region_model,
             refinement_pass)));
                                Eigen::Matrix<double, 2, 3> J1;
                                J1 << 1.0, 0.0, dealty - dy0,
                                    0.0, 1.0, -dealtx - dx0;
                                J = J0.transpose() * J1;
                            }
                            else if (iter == 0)
                            {
                                // double dx, dy;
                                // dx = g1 + (kp[m].y - kp[0].y) * g3 + (kp[m].x - kp[0].x) * g4;
                                // dy = g2 - (kp[m].x - kp[0].x) * g3 + (kp[m].y - kp[0].y) * g4;

                                J0 = -1.0 * Eigen::Vector2d(
    0.5 *
        (GetPhotometricallyCorrectedPixelValue(
             img2,
             kp[m].x + dx + x + 1,
             kp[m].y + dy + y,
             region_model,
             refinement_pass) -
         GetPhotometricallyCorrectedPixelValue(
             img2,
             kp[m].x + dx + x - 1,
             kp[m].y + dy + y,
             region_model,
             refinement_pass)),
    0.5 *
        (GetPhotometricallyCorrectedPixelValue(
             img2,
             kp[m].x + dx + x,
             kp[m].y + dy + y + 1,
             region_model,
             refinement_pass) -
         GetPhotometricallyCorrectedPixelValue(
             img2,
             kp[m].x + dx + x,
             kp[m].y + dy + y - 1,
             region_model,
             refinement_pass)));
                                Eigen::Matrix<double, 2, 3> J1;
                                J1 << 1.0, 0.0, dealty - dy0,
                                    0.0, 1.0, -dealtx - dx0;
                                J = J0.transpose() * J1;
                            }

                            // compute H, b and set cost;
                            b += -error * J;
                            // cost += huberloss(error, 1) * 2;
                            cost += error * error;
                            if (inverse == false || iter == 0)
                            {
                                // also update H
                                H += J * J.transpose();
                            }
                        }
                    // if (iter == 0)
                    errorGray += abs(gray1 - gray2) / ((2 * half_patch_size + 1) * (2 * half_patch_size + 1));
                }
                chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
                auto time_used1 = chrono::duration_cast<chrono::duration<double>>(t2 - t1);

                chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
                // compute update
                Eigen::Vector3d update = H.ldlt().solve(b);
                chrono::steady_clock::time_point t4 = chrono::steady_clock::now();

                auto time_used2 = chrono::duration_cast<chrono::duration<double>>(t4 - t3);
                // ROS_WARN("compute time DIV solve time : %f", float(time_used1.count() / time_used2.count()));

                if (std::isnan(update[0]) || std::isnan(update[1]) || std::isnan(update[2]) ||
                    std::isinf(update[0]) || std::isinf(update[1]) || std::isinf(update[2]))
                {
                    succ = -1;
                    ROS_WARN("fail line \n");
                    break;
                }

                if (iter > 0 && cost > lastCost)
                {
                    break;
                }

                // update dx, dy
                g1 += update[0];
                g2 += update[1];
                g3 += update[2];

                lastCost = cost;
                succ = i;

                if (update.norm() < 1e-2)
                {
                    // converge
                    // ROS_WARN("converge\n");
                    break;
                }
            }
            cost /= ((2 * half_patch_size + 1) * (2 * half_patch_size + 1));

            // ROS_WARN("Error:%f  AvrError:%f", cost, errorGray);
            if (layer == 0 &&
    errorGray >= Graydoor * _POINTNUM_)
{
    gray_rejected = true;
    succ = -1;
}
            // else if (layer == 0)
            // {
            //     ofstream fout("/home/shitong/study/slam/lfvins_catkin/src/linefeature_tracker/src/error.csv", ios::out | ios::app);
            //     fout << cost << ',' << errorGray << endl;
            //     fout.close();
            // }

            // set kp2
            // if (layer == 0)
            for (int m = 0; m < kp.size(); m++)
            {
                //挨个存入kp2
                float dx = g1 + (kp[m].y - kp[0].y) * g3;
                float dy = g2 - (kp[m].x - kp[0].x) * g3;
                float x = kp[m].x + dx;
                float y = kp[m].y + dy;

                if (m == 0 && (x <= 0 || x >= img1.cols - 1 ||
                               y <= 0 || y >= img1.rows - 1))
                    succ = -1;

                if (kp2[i].keyPoint.size() == _POINTNUM_)
                    kp2[i].keyPoint[m] = (Point2f(x, y));
                else
                    kp2[i].keyPoint.push_back(Point2f(x, y));
            }

            // if (layer == 0 && checkgoodLine(img2, kp2[i].keyPoint, 10, 1) == false)
            //     succ = -1;
            if (layer == 0 && checkgoodLine(magnitude, kp2[i].keyPoint) == false)
                succ = -1;

            success[i] = succ;
            if (layer == 0 && record_diagnostics)
{
    const double point_count =
        kp.empty() ? 1.0 : static_cast<double>(kp.size());

    const Point2f diagnostic_point =
        kp.empty() ? Point2f(-1.0f, -1.0f) : kp[0];

    std::lock_guard<std::mutex> lock(g_line_diag_mutex);

    if (g_line_diag_file.is_open())
    {
        g_line_diag_file
            << g_line_diag_frame << ","
            << i << ","
            << diagnostic_point.x << ","
            << diagnostic_point.y << ","
            << cost / point_count << ","
            << errorGray / point_count << ","
            << (gray_rejected ? 1 : 0) << ","
            << (succ != -1 ? 1 : 0) << ","
            << g1 << ","
            << g2 << ","
            << g3 << "\n";
    }
}
        }
    }
}

namespace
{

struct PhotometricAccumulator
{
    int count = 0;
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_xx = 0.0;
    double sum_xy = 0.0;
    double sum_yy = 0.0;

    void add(double current_gray, double previous_gray)
    {
        ++count;
        sum_x += current_gray;
        sum_y += previous_gray;
        sum_xx += current_gray * current_gray;
        sum_xy += current_gray * previous_gray;
        sum_yy += previous_gray * previous_gray;
    }
};

bool solveAffinePhotometricModel(
    const PhotometricAccumulator &accumulator,
    int minimum_samples,
    double &gain,
    double &bias)
{
    if (accumulator.count < minimum_samples)
        return false;

    const double count =
        static_cast<double>(accumulator.count);

    const double mean_x = accumulator.sum_x / count;
    const double mean_y = accumulator.sum_y / count;

    const double variance_x =
        accumulator.sum_xx / count - mean_x * mean_x;

    const double covariance_xy =
        accumulator.sum_xy / count - mean_x * mean_y;

    // 区域纹理或灰度变化太小，无法稳定估计增益。
    if (variance_x < 25.0)
        return false;

    gain = covariance_xy / variance_x;
    bias = mean_y - gain * mean_x;

    if (!std::isfinite(gain) || !std::isfinite(bias))
        return false;

    // 防止少量错误匹配产生极端参数。
    gain = std::max(0.5, std::min(2.0, gain));
    bias = std::max(-50.0, std::min(50.0, bias));

    return true;
}

double computePhotometricMse(
    const PhotometricAccumulator &accumulator,
    double gain,
    double bias)
{
    if (accumulator.count <= 0)
        return 0.0;

    const double count =
        static_cast<double>(accumulator.count);

    // sum((previous - gain * current - bias)^2)
    double squared_error =
        accumulator.sum_yy +
        gain * gain * accumulator.sum_xx +
        count * bias * bias -
        2.0 * gain * accumulator.sum_xy -
        2.0 * bias * accumulator.sum_y +
        2.0 * gain * bias * accumulator.sum_x;

    // 浮点舍入可能产生极小的负数。
    squared_error = std::max(0.0, squared_error);

    return squared_error / count;
}

void estimateRegionPhotometricModel(
    const cv::Mat &previous_image,
    const cv::Mat &current_image,
    const vector<Line> &previous_lines,
    const vector<Line> &current_lines,
    const vector<int> &tracking_status,
    RegionPhotometricModel &region_model)
{

    region_model.reset(current_image.cols, current_image.rows);

    if (previous_image.empty() || current_image.empty())
        return;

    std::array<
        PhotometricAccumulator,
        RegionPhotometricModel::kRegionCount>
        regional_accumulators;

    PhotometricAccumulator global_accumulator;

    const int patch_radius = 2;
const int minimum_region_samples = 150;
const int minimum_region_lines = 2;

    const size_t line_count = std::min(
        previous_lines.size(),
        std::min(current_lines.size(), tracking_status.size()));

    for (size_t line_index = 0;
         line_index < line_count;
         ++line_index)
    {
        if (tracking_status[line_index] == -1)
            continue;

            std::array<int, RegionPhotometricModel::kRegionCount>
    line_region_flags;

line_region_flags.fill(0);

        const vector<Point2f> &previous_points =
            previous_lines[line_index].keyPoint;

        const vector<Point2f> &current_points =
            current_lines[line_index].keyPoint;

        const size_t point_count =
            std::min(previous_points.size(), current_points.size());

        for (size_t point_index = 0;
             point_index < point_count;
             ++point_index)
        {
            const Point2f &previous_point =
                previous_points[point_index];

            const Point2f &current_point =
                current_points[point_index];

            for (int offset_y = -patch_radius;
                 offset_y <= patch_radius;
                 ++offset_y)
            {
                for (int offset_x = -patch_radius;
                     offset_x <= patch_radius;
                     ++offset_x)
                {
                    const float previous_x =
                        previous_point.x + offset_x;
                    const float previous_y =
                        previous_point.y + offset_y;

                    const float current_x =
                        current_point.x + offset_x;
                    const float current_y =
                        current_point.y + offset_y;

                    // GetPixelValue使用双线性插值，需要为右下像素留边界。
                    if (previous_x < 1.0f ||
                        previous_x >= previous_image.cols - 2 ||
                        previous_y < 1.0f ||
                        previous_y >= previous_image.rows - 2 ||
                        current_x < 1.0f ||
                        current_x >= current_image.cols - 2 ||
                        current_y < 1.0f ||
                        current_y >= current_image.rows - 2)
                    {
                        continue;
                    }

                    const double previous_gray =
                        GetPixelValue(
                            previous_image,
                            previous_x,
                            previous_y);

                    const double current_gray =
                        GetPixelValue(
                            current_image,
                            current_x,
                            current_y);

                    // 排除接近饱和或纯黑的像素。
                    if (previous_gray <= 5.0 ||
                        previous_gray >= 250.0 ||
                        current_gray <= 5.0 ||
                        current_gray >= 250.0)
                    {
                        continue;
                    }

                    const int region_index =
    region_model.regionIndex(
        current_x,
        current_y);

line_region_flags[region_index] = 1;

regional_accumulators[region_index].add(
                        current_gray,
                        previous_gray);

                                    global_accumulator.add(
                    current_gray,
                    previous_gray);
            }
        }
    }

    // 当前线在每个区域最多计数一次。
    for (int region_index = 0;
         region_index < RegionPhotometricModel::kRegionCount;
         ++region_index)
    {
        if (line_region_flags[region_index])
            ++region_model.line_count[region_index];
    }
}

    double global_gain = 1.0;
    double global_bias = 0.0;

    const bool global_valid =
        solveAffinePhotometricModel(
            global_accumulator,
            minimum_region_samples,
            global_gain,
            global_bias);

    for (int region_index = 0;
         region_index < RegionPhotometricModel::kRegionCount;
         ++region_index)
    {
        region_model.sample_count[region_index] =
            regional_accumulators[region_index].count;

        double region_gain = 1.0;
        double region_bias = 0.0;

        const bool region_valid =
    region_model.line_count[region_index] >=
        minimum_region_lines &&
    solveAffinePhotometricModel(
        regional_accumulators[region_index],
        minimum_region_samples,
        region_gain,
        region_bias);

        if (region_valid)
{
    region_model.gain[region_index] = region_gain;
    region_model.bias[region_index] = region_bias;
    region_model.valid[region_index] = 1;

    const double raw_region_mse =
        computePhotometricMse(
            regional_accumulators[region_index],
            1.0,
            0.0);

    const double fitted_region_mse =
        computePhotometricMse(
            regional_accumulators[region_index],
            region_gain,
            region_bias);

    double relative_improvement = 0.0;

    if (raw_region_mse > 1e-9)
    {
        relative_improvement =
            (raw_region_mse - fitted_region_mse) /
            raw_region_mse;
    }

    relative_improvement =
        std::max(0.0, relative_improvement);

    const double pixel_confidence =
        std::min(
            1.0,
            region_model.sample_count[region_index] / 300.0);

    const double line_confidence =
        std::min(
            1.0,
            region_model.line_count[region_index] / 4.0);

    const double improvement_confidence =
        std::min(
            1.0,
            relative_improvement / 0.10);

    region_model.confidence[region_index] =
        pixel_confidence *
        line_confidence *
        improvement_confidence;
}
else
{
            // 区域样本不足时回退到全图模型；全图也无效则保持1和0。
            region_model.gain[region_index] =
                global_valid ? global_gain : 1.0;

            region_model.bias[region_index] =
                global_valid ? global_bias : 0.0;

            region_model.valid[region_index] = 0;
            region_model.confidence[region_index] = 0.0;
        }
    }

    // 对有效区域做一次轻量邻域平滑。
    const std::array<
        double,
        RegionPhotometricModel::kRegionCount>
        original_gain = region_model.gain;

    const std::array<
        double,
        RegionPhotometricModel::kRegionCount>
        original_bias = region_model.bias;

    for (int row = 0;
         row < RegionPhotometricModel::kRows;
         ++row)
    {
        for (int col = 0;
             col < RegionPhotometricModel::kCols;
             ++col)
        {
            const int center_index =
                region_model.index(row, col);

            if (!region_model.valid[center_index])
                continue;

            double neighbor_gain_sum = 0.0;
            double neighbor_bias_sum = 0.0;
            int neighbor_count = 0;

            const int neighbor_rows[4] =
                {row - 1, row + 1, row, row};

            const int neighbor_cols[4] =
                {col, col, col - 1, col + 1};

            for (int neighbor = 0; neighbor < 4; ++neighbor)
            {
                const int neighbor_row = neighbor_rows[neighbor];
                const int neighbor_col = neighbor_cols[neighbor];

                if (neighbor_row < 0 ||
                    neighbor_row >= RegionPhotometricModel::kRows ||
                    neighbor_col < 0 ||
                    neighbor_col >= RegionPhotometricModel::kCols)
                {
                    continue;
                }

                const int neighbor_index =
                    region_model.index(
                        neighbor_row,
                        neighbor_col);

                if (!region_model.valid[neighbor_index])
                    continue;

                neighbor_gain_sum +=
                    original_gain[neighbor_index];

                neighbor_bias_sum +=
                    original_bias[neighbor_index];

                ++neighbor_count;
            }

            if (neighbor_count > 0)
            {
                const double mean_neighbor_gain =
                    neighbor_gain_sum / neighbor_count;

                const double mean_neighbor_bias =
                    neighbor_bias_sum / neighbor_count;

                region_model.gain[center_index] =
                    0.75 * original_gain[center_index] +
                    0.25 * mean_neighbor_gain;

                region_model.bias[center_index] =
                    0.75 * original_bias[center_index] +
                    0.25 * mean_neighbor_bias;
            }
        }
    }

    // 记录区域模型，但暂时不用于光流残差。
    static std::ofstream region_log_file;
    static int region_frame_index = 0;

    if (!region_log_file.is_open())
    {
        region_log_file.open(
            "/tmp/eplf_region_model.csv",
            std::ios::out | std::ios::trunc);

        region_log_file
    << "frame,row,col,gain,bias,"
    << "sample_count,line_count,valid,confidence,"
    << "raw_mse,corrected_mse\n";
    }

    if (region_log_file.is_open())
    {
        for (int row = 0;
             row < RegionPhotometricModel::kRows;
             ++row)
        {
            for (int col = 0;
                 col < RegionPhotometricModel::kCols;
                 ++col)
            {
                const int region_index =
                    region_model.index(row, col);

                    const double raw_mse =
    computePhotometricMse(
        regional_accumulators[region_index],
        1.0,
        0.0);

const double corrected_mse =
    computePhotometricMse(
        regional_accumulators[region_index],
        region_model.gain[region_index],
        region_model.bias[region_index]);

                region_log_file
                    << region_frame_index << ","
                    << row << ","
                    << col << ","
                    << region_model.gain[region_index] << ","
                    << region_model.bias[region_index] << ","
                    << region_model.sample_count[region_index] << ","
                    << region_model.line_count[region_index] << ","
                    << region_model.valid[region_index] << ","
                    << region_model.confidence[region_index] << ","
                    << raw_mse << ","
                    << corrected_mse << "\n";
            }
        }

        region_log_file.flush();
    }

    ++region_frame_index;
}

} // namespace


void OpticalFlowMultiLevel(
    const Mat &magnitude,
    const Mat &angle,
    vector<Mat> &img1_pyr,
    vector<Mat> &img2_pyr,
    const vector<Line> &kp1,
    vector<Line> &kp2,
    vector<int> &success,
    RegionPhotometricModel &region_model,
    bool inverse)
{
    // parameters
    int pyramids = _LAYER_;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};
    vector<int> successd{0};
    param.resize(kp1.size());

    // create pyramids

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    if (img1_pyr.size() == 1)
        for (int i = 1; i < pyramids; i++)
        {

            Mat img1;
            cv::resize(img1_pyr[i - 1], img1,
                       cv::Size(img1_pyr[i - 1].cols * pyramid_scale, img1_pyr[i - 1].rows * pyramid_scale));
            img1_pyr.push_back(img1);
        }
    if (img2_pyr.size() == 1)
        for (int i = 1; i < pyramids; i++)
        {
            Mat img2;
            cv::resize(img2_pyr[i - 1], img2,
                       cv::Size(img2_pyr[i - 1].cols * pyramid_scale, img2_pyr[i - 1].rows * pyramid_scale));
            img2_pyr.push_back(img2);
        }
    if (img1_pyr.size() <= 1 || img2_pyr.size() <= 1)
        ROS_WARN("error img_pyr imput\n");
    // coarse-to-fine LK tracking in pyramids
    vector<Line> kp1_pyr, kp2_pyr;
    for (auto &kp : kp1)
    {
        auto kp_top = kp;
        for (int m = 0; m < kp_top.keyPoint.size(); m++)
        {
            kp_top.keyPoint[m] *= scales[pyramids - 1];
        }
        kp1_pyr.push_back(kp_top);
        // kp2_pyr.push_back(kp_top);
    }
    // ROS_WARN("multi layer img\n");

    for (int level = pyramids - 1; level >= 0; level--)
    {
        // from coarse to fine
        successd.clear();

        // ROS_WARN("begin single level(%d)\n", level);
        chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
        if (level == 0)
            OpticalFlowSingleLevel(magnitude, img1_pyr[level], img2_pyr[level], kp1_pyr, kp2_pyr, successd, false, true, level);
        else
            OpticalFlowSingleLevel(magnitude, img1_pyr[level], img2_pyr[level], kp1_pyr, kp2_pyr, successd, false, true, level);
        // ROS_WARN("success single level(%d)\n", level);
        chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
        auto time_used = chrono::duration_cast<chrono::duration<double>>(t4 - t3);
        // ROS_WARN("layer : %d  use %f s\n", level, time_used);

        if (level > 0)
        {
            for (auto &kp : kp1_pyr)
                // cout << "size:" << kp.size() << endl;
                for (int m = 0; m < kp.keyPoint.size(); m++)
                {
                    kp.keyPoint[m] /= pyramid_scale;
                }
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    // ROS_WARN("line : %d  use %f s\n", kp1.size(), time_used);
    // ROS_WARN("success multi level, %d lines\n", kp2_pyr.size());
estimateRegionPhotometricModel(
    img1_pyr[0],
    img2_pyr[0],
    kp1_pyr,
    kp2_pyr,
    successd,
    region_model);

    //在kp2中，真正留下的是追踪成功的点
    //这些点对应的序号在success中保存
    cv::Mat mergel = cv::Mat(ROW, COL, CV_8UC1, 255);
    cv::Mat maskl = cv::Mat(ROW, COL, CV_8UC1, 255);
    int lineIndex = 0;
    for (int m = 0; m < kp2_pyr.size(); m++)
    {
        if (successd[m] != -1)
        {
            kp2_pyr[m].recoverLine(successd[m], magnitude, angle, 5);

            bool ifMerge = false;
            // ROS_WARN("adding line ...");
            if (mergel.at<uchar>(kp2_pyr[m].MidPt) != 255 ||
                mergel.at<uchar>(kp2_pyr[m].StartPt) != 255 ||
                mergel.at<uchar>(kp2_pyr[m].EndPt) != 255)
            {
                // ROS_WARN("merge tracked line!!!\n");
                ifMerge = mergeLine(kp2_pyr[m], mergel, kp2, maskl);
                // ROS_WARN("success add line\n");
            }
            if (!ifMerge)
            {
                kp2.push_back(kp2_pyr[m]);
                success.push_back(m);
                drawLine(kp2_pyr[m], mergel, lineIndex++);
            }
        }
    }
}
#endif

bool checkBoundry(const Point2f &point, int row, int col)
{
    if (point.x >= 0 && point.x < col && point.y >= 0 && point.y < row)
        return true;
    return false;
}

void drawLine(const Line &line, Mat &merge, int index)
{
    float lenthThreshod = 20;

    //* 画线，贯穿图像的线，给每条线占位置，另一条线进入就能判断共线
    if (index > 255)
    {
        ROS_WARN("index > 255  : %d\n", index);
        return;
    }
    // ROS_WARN("drawing line...");

    int row = merge.rows - 1;
    int col = merge.cols - 1;
    Point2f start = line.StartPt;
    Point2f end = line.EndPt;
    Point2f drawStart, drawEnd;

    float length = sqrt((start.x - end.x) * (start.x - end.x) + (start.y - end.y) * (start.y - end.y));
    float dealtx = abs(start.x - end.x) / length * lenthThreshod;
    float dealty = abs(start.y - end.y) / length * lenthThreshod;

    //判断start 和 end 在 x 方向的大小
    if (start.x > end.x)
    {
        float extendStart = abs(col - start.x) > dealtx ? (col - start.x) / abs(col - start.x) * dealtx : (col - start.x);
        float extendEnd = abs(0 - end.x) > dealtx ? (0 - end.x) / abs(0 - end.x) * dealtx : (0 - end.x);

        drawStart = start + (extendStart) / (start.x - end.x) * (start - end);
        drawEnd = end + (float)(extendEnd) / (end.x - start.x) * (end - start);
    }
    else if (start.x < end.x)
    {
        float extendStart = abs(0 - start.x) > dealtx ? (0 - start.x) / abs(0 - start.x) * dealtx : (0 - start.x);
        float extendEnd = abs(col - end.x) > dealtx ? (col - end.x) / abs(col - end.x) * dealtx : (col - end.x);

        drawStart = start + (float)(extendStart) / (start.x - end.x) * (start - end);
        drawEnd = end + (float)(extendEnd) / (end.x - start.x) * (end - start);
    }
    else
    {
        drawStart = Point2f(start.x, 0);
        drawEnd = Point2f(start.x, row);
    }

    //检查y超出边界了没
    if (drawStart.y > row)
        drawStart = start + (float)(row - start.y) / (start.y - end.y) * (start - end);
    else if (drawStart.y < 0)
        drawStart = start + (float)(0 - start.y) / (start.y - end.y) * (start - end);

    if (drawEnd.y > row)
        drawEnd = end + (float)(row - end.y) / (end.y - start.y) * (end - start);
    else if (drawEnd.y < 0)
        drawEnd = end + (float)(0 - end.y) / (end.y - start.y) * (end - start);

    // draw line
    cv::line(merge, drawStart, drawEnd, index, 4);
    // imshow("merge", merge);
    // waitKey();
    // ROS_WARN("success\n");
}
bool mergeLine(const Line &line, const Mat &merge, vector<Line> &vecline, Mat &maskl)
{
    //* 短线合并成长线，写的比较笨，但是能用
    float angleThreshold = 1;

    float thisAngle = atan2(line.StartPt.y - line.EndPt.y, line.StartPt.x - line.EndPt.x) * 180 / M_PI;
    if (thisAngle < 0)
        thisAngle += 180;

    int g[3];
    if (!checkBoundry(line.StartPt, ROW, COL) || !checkBoundry(line.MidPt, ROW, COL) || !checkBoundry(line.EndPt, ROW, COL))
        return true;
    g[0] = merge.at<uchar>(line.StartPt);
    g[1] = merge.at<uchar>(line.MidPt);
    g[2] = merge.at<uchar>(line.EndPt);

    int lineIndex = 0;
    for (int i = 0; i < 3; i++)
    {
        lineIndex = g[i];
        if (lineIndex > vecline.size())
        {
            lineIndex = -1;
            continue;
        }
        Point2f &mergeStart = vecline[lineIndex].StartPt;
        Point2f &mergeEnd = vecline[lineIndex].EndPt;

        float mergeAngle = atan2(mergeEnd.y - mergeStart.y, mergeEnd.x - mergeStart.x) * 180 / M_PI;
        if (mergeAngle < 0)
            mergeAngle += 180;

        if (abs(mergeAngle - thisAngle) < angleThreshold ||
            abs(180 - abs(mergeAngle - thisAngle)) < angleThreshold)
            break;

        lineIndex = -1;
    }

    if (lineIndex == -1 || lineIndex > vecline.size())
        return false;

    // ROS_WARN("mergeing line ..");

    // cv::Mat showMerge = cv::Mat(ROW, COL, CV_8UC1, 255);
    // cv::line(showMerge, vecline[lineIndex].StartPt, vecline[lineIndex].EndPt, 50, 1);

    Point2f &mergeStart = vecline[lineIndex].StartPt;
    Point2f &mergeEnd = vecline[lineIndex].EndPt;

    if (thisAngle < 45 || thisAngle > 135)
    {
        if (mergeStart.x > mergeEnd.x)
        {
            if (line.StartPt.x > mergeStart.x)
                mergeStart = line.StartPt;
            else if (line.StartPt.x < mergeEnd.x)
                mergeEnd = line.StartPt;
            if (line.EndPt.x > mergeStart.x)
                mergeStart = line.EndPt;
            else if (line.EndPt.x < mergeEnd.x)
                mergeEnd = line.EndPt;
        }
        else if (mergeStart.x < mergeEnd.x)
        {
            if (line.StartPt.x < mergeStart.x)
                mergeStart = line.StartPt;
            else if (line.StartPt.x > mergeEnd.x)
                mergeEnd = line.StartPt;
            if (line.EndPt.x < mergeStart.x)
                mergeStart = line.EndPt;
            else if (line.EndPt.x > mergeEnd.x)
                mergeEnd = line.EndPt;
        }
        else
        {
            vector<float> yArr = {mergeStart.y, mergeEnd.y, line.StartPt.y, line.EndPt.y};
            sort(yArr.begin(), yArr.end());
            mergeStart = Point2f(mergeStart.x, yArr[0]);
            mergeEnd = Point2f(mergeEnd.x, yArr[yArr.size() - 1]);
        }
    }
    else
    {
        if (mergeStart.y > mergeEnd.y)
        {
            if (line.StartPt.y > mergeStart.y)
                mergeStart = line.StartPt;
            else if (line.StartPt.y < mergeEnd.y)
                mergeEnd = line.StartPt;
            if (line.EndPt.y > mergeStart.y)
                mergeStart = line.EndPt;
            else if (line.EndPt.y < mergeEnd.y)
                mergeEnd = line.EndPt;
        }
        else if (mergeStart.y < mergeEnd.y)
        {
            if (line.StartPt.y < mergeStart.y)
                mergeStart = line.StartPt;
            else if (line.StartPt.y > mergeEnd.y)
                mergeEnd = line.StartPt;
            if (line.EndPt.y < mergeStart.y)
                mergeStart = line.EndPt;
            else if (line.EndPt.y > mergeEnd.y)
                mergeEnd = line.EndPt;
        }
        else
        {
            vector<float> yArr = {mergeStart.x, mergeEnd.x, line.StartPt.x, line.EndPt.x};
            sort(yArr.begin(), yArr.end());
            mergeStart = Point2f(yArr[0], mergeStart.y);
            mergeEnd = Point2f(yArr[yArr.size() - 1], mergeEnd.y);
        }
    }

    vecline[lineIndex].resetLine();
    cv::line(maskl, mergeStart, mergeEnd, 0, 30);
    // cv::line(showMerge, vecline[lineIndex].StartPt, vecline[lineIndex].EndPt, 100, 1);
    // imshow("showMerge", showMerge);
    // waitKey();

    // ROS_WARN("success\n");
    return true;
}
bool mergeTrackedLine(const Line &line, const Mat &merge, vector<Line> &vecline, Mat &mergel)
{
    //* 用提取到的线来替代追踪到的线条
    //! 我难以解释原因，但线流是有一定优势的，这个函数没被用到
    int g1, g2, g3;
    g1 = merge.at<uchar>(line.StartPt);
    g2 = merge.at<uchar>(line.MidPt);
    g3 = merge.at<uchar>(line.EndPt);
    int lineIndex = (g1 == g2 || g1 == g3) ? g1 : (g2 == g3 ? g2 : -1);

    if (lineIndex == -1 || lineIndex > vecline.size())
        return false;
    Point2f &mergeStart = vecline[lineIndex].StartPt;
    Point2f &mergeEnd = vecline[lineIndex].EndPt;
    Point2f &mergeMid = vecline[lineIndex].MidPt;

    float mergeAngle = atan2(mergeEnd.y - mergeStart.y, mergeEnd.x - mergeStart.x) * 180 / M_PI;
    if (mergeAngle < 180)
        mergeAngle += 180;
    float thisAngle = atan2(line.StartPt.y - line.EndPt.y, line.StartPt.x - line.EndPt.x) * 180 / M_PI;
    if (thisAngle < 180)
        thisAngle += 180;

    float angleThreshold = 1;
    if (abs(mergeAngle - thisAngle) > angleThreshold &&
        abs(180 - abs(mergeAngle - thisAngle)) > angleThreshold)
        return false;

    // if (sqrt((mergeMid.x - line.MidPt.x) * (mergeMid.x - line.MidPt.x) + (mergeMid.y - line.MidPt.y) * (mergeMid.y - line.MidPt.y)) < 10)
    // {
    //     ROS_WARN("!mergeing tracked line ..\n");
    //     mergeStart = line.StartPt;
    //     mergeEnd = line.EndPt;
    //     vecline[lineIndex].resetLine();
    //     cv::line(mergel, mergeStart, mergeEnd, lineIndex, 3);
    //     return true;
    // }
    if (line.StartPt.x > line.EndPt.x)
        if (mergeStart.x < line.StartPt.x && mergeStart.x > line.EndPt.x &&
            mergeEnd.x < line.StartPt.x && mergeEnd.x > line.EndPt.x)
        {
            ROS_WARN("!mergeing tracked line ..\n");
            mergeStart = line.StartPt;
            mergeEnd = line.EndPt;
            vecline[lineIndex].resetLine();
            cv::line(mergel, mergeStart, mergeEnd, lineIndex, 3);

            return true;
        }
    if (line.StartPt.x < line.EndPt.x)
        if (mergeStart.x > line.StartPt.x && mergeStart.x < line.EndPt.x &&
            mergeEnd.x > line.StartPt.x && mergeEnd.x < line.EndPt.x)
        {
            ROS_WARN("!mergeing tracked line ..\n");
            mergeStart = line.StartPt;
            mergeEnd = line.EndPt;
            vecline[lineIndex].resetLine();
            cv::line(mergel, mergeStart, mergeEnd, lineIndex, 3);
            return true;
        }

    return false;
}

void LineFeatureTracker::readImage(const cv::Mat &_img)
{

    cv::Mat photometric_img;
TicToc t_p;
frame_cnt++;

// 先得到不经过CLAHE的去畸变图像
cv::remap(
    _img,
    photometric_img,
    undist_map1_,
    undist_map2_,
    CV_INTER_LINEAR);

// 复制一份用于线检测和梯度计算
cv::Mat img = photometric_img.clone();

if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(img, img);
    }
    chrono::steady_clock::time_point t01 = chrono::steady_clock::now();
    bool first_img = false;
    if (forw_img == nullptr) // 系统初始化的第一帧图像
    {
        forw_img.reset(new FrameLines);
        cur_img.reset(new FrameLines);
        forw_img->img = img;
forw_img->img_pyr.push_back(img);
forw_img->photometric_img = photometric_img;
forw_img->photometric_img_pyr.push_back(photometric_img);

cur_img->img = img;
cur_img->img_pyr.push_back(img);
cur_img->photometric_img = photometric_img;
cur_img->photometric_img_pyr.push_back(photometric_img);
        first_img = true;
        // ROS_WARN("compute gradient ...  ");
        Mat x_arr, y_arr;
        Sobel(img, x_arr, CV_32F, 1, 0);
        Sobel(img, y_arr, CV_32F, 0, 1);
        cartToPolar(x_arr, y_arr, forw_img->magnitude, forw_img->angle, true);
        // ROS_WARN("success: %d \n ", forw_img->magnitude.rows);
    }
    else
    {
        forw_img.reset(new FrameLines); // 初始化一个新的帧
        forw_img->img = img;
forw_img->img_pyr.push_back(img);
forw_img->photometric_img = photometric_img;
forw_img->photometric_img_pyr.push_back(photometric_img);
        // ROS_WARN("compute gradient ...  ");
        Mat x_arr, y_arr;
        // Mat x_arr0,y_arr0;
        Sobel(img, x_arr, CV_32F, 1, 0);
        Sobel(img, y_arr, CV_32F, 0, 1);
        cartToPolar(x_arr, y_arr, forw_img->magnitude, forw_img->angle, true);

        // ROS_WARN("success: %d \n ", forw_img->magnitude.rows);
        // imshow("x_arr", x_arr);
        // imshow("y_arr", y_arr);
        // imshow("mag", forw_img->magnitude);
        // imshow("angle", forw_img->angle);
        // waitKey();
    }
    chrono::steady_clock::time_point t02 = chrono::steady_clock::now();
    auto time_used0 = chrono::duration_cast<chrono::duration<double>>(t02 - t01);

    int linenum = 40;
    int addLineNum = 10;
    float ratio = 0;

    double min_edline_length = 0.125;

    Ptr<cv::ximgproc::EdgeDrawing> ed = cv::ximgproc::createEdgeDrawing();
    ed->params.EdgeDetectionOperator = cv::ximgproc::EdgeDrawing::SOBEL;
    ed->params.GradientThresholdValue = 32;
    ed->params.AnchorThresholdValue = 16;
    ed->params.ScanInterval = 2;
    ed->params.MinLineLength = min_edline_length * (std::min(img.cols, img.rows));
    vector<Vec4f> edlines, new_edlines;

    // int LINELENGTH;
    float LINELENGTH = 0;

#if _SHITONG_
    if (first_img == true || cur_img->vecLine.empty())
    {
        //第一帧提取
        ROS_WARN("empty");
        
        ed->detectEdges(img);ed->detectLines(edlines);
        if (edlines.size() < linenum + addLineNum)
        {
            ed->params.AnchorThresholdValue *= 0.5;
            ed->params.ScanInterval *= 0.5;
            
            ed->detectEdges(img);ed->detectLines(edlines);
        }
        cv::Mat maskl = cv::Mat(ROW, COL, CV_8UC1, 255);
        cv::Mat mergel = cv::Mat(ROW, COL, CV_8UC1, 255);

        // {
        //     vector<float> lineLength;

        //     for (auto line : edlines)
        //         lineLength.push_back(sqrt((line[0] - line[2]) * (line[0] - line[2]) + (line[1] - line[3]) * (line[1] - line[3])));
        //     sort(lineLength.begin(), lineLength.end());
        //     if (!lineLength.empty())
        //         // LINELENGTH = lineLength[max(int(lineLength.size() * ratio), (int)lineLength.size() - 40)];
        //         LINELENGTH = lineLength[int(lineLength.size() * ratio)];

        //     else
        //         LINELENGTH = 0;
        // }
        int num = linenum + addLineNum;
        // ROS_WARN("line Length: %f", LINELENGTH);
        int lineIndex = 0;
        for (auto line : edlines)
        {
            Line l(line);
            bool ifMerge = false;
#if _CONTROLLINEDENSE_
            if ((mergel.at<uchar>(l.MidPt) != 255 ||
                 mergel.at<uchar>(l.StartPt) != 255 ||
                 mergel.at<uchar>(l.EndPt) != 255) &&
                (ifMerge = mergeLine(l, mergel, forw_img->vecLine, maskl)))
                ;
            else if (maskl.at<uchar>(l.MidPt) != 255 ||
                     num <= 0)
                continue;

            if (ifMerge)
                continue;
#endif
            // if (l.length >= LINELENGTH && goodLine(img, line))
            if (l.length >= LINELENGTH)
            {

                forw_img->vecLine.push_back(l);
                forw_img->success.push_back(-1);
                forw_img->lineID.push_back(allfeature_cnt++);

#if _SHOWLINEFLOW_

                LineST linest;
                linest.start = l.StartPt;
                linest.end = l.EndPt;

                LineRecord linerecord;
                linerecord.push_back(linest);

                forw_img->lineRec.push_back(linerecord);

#endif
#if _ONELINE_
                break;
#endif
                num--;
                // if (num <= 0)
                //     break;
#if _CONTROLLINEDENSE_
                cv::line(maskl, l.StartPt, l.EndPt, 0, 30);
                drawLine(l, mergel, lineIndex++);
#endif
            }
        }
    }
    else
    {

        {
            //追踪
            // ROS_WARN("-> here : %d\n", cur_img->vecLine.size());
            TicToc t_lineflow;
           OpticalFlowMultiLevel(
    forw_img->magnitude,
    forw_img->angle,
    cur_img->photometric_img_pyr,
    forw_img->photometric_img_pyr,
    cur_img->vecLine,
    forw_img->vecLine,
    forw_img->success,
    forw_img->region_photometric_model);
            double lineflowtime = t_lineflow.toc() ;
            ofstream fout("/home/jiangdi/result_output/time/euroc/EPLF_VINS_WS/eplfvins_line_tracking.csv", ofstream::app);
            fout <<lineflowtime <<endl;
            fout.close();
            TicToc t_linedetect;

            //把追到的线先存下来
            for (int i = 0; i < forw_img->vecLine.size() && i < 255; i++)
            {
                //存下来追踪到的线的序号和追踪次数
                if (forw_img->success[i] >= 0)
                {
                    // ROS_WARN("push to vecLine (%d)\n", (forw_img->success[i]));
                    forw_img->lineID.push_back(cur_img->lineID[forw_img->success[i]]);
#if _SHOWLINEFLOW_
                    //原来的追踪同步到这里
                    forw_img->lineRec.push_back(cur_img->lineRec[forw_img->success[i]]);
                    LineST linest;
                    linest.start = forw_img->vecLine[i].StartPt;
                    linest.end = forw_img->vecLine[i].EndPt;
                    //加上当前的跟踪
                    forw_img->lineRec[i].push_back(linest);
#endif
                }
            }
            // ROS_WARN("push to vecLine (%f)\n", allfeature_cnt);

            //线少于一定数，再补充线
            if (forw_img->vecLine.size() < linenum - 5)
            {
                cv::Mat maskl = cv::Mat(ROW, COL, CV_8UC1, 255);
                cv::Mat mergel = cv::Mat(ROW, COL, CV_8UC1, 255);
                setMask();
                setMerge(mergel);





                //提取新的
                // lsd_->detect(img, newlsd, 2, 1, opts, mask);
                
                ed->detectEdges(img);ed->detectLines(new_edlines);
                /*if (new_edlines.size() < linenum + addLineNum)
                {
                    ed->params.AnchorThresholdValue *= 0.5;
                    ed->params.ScanInterval *= 0.5;
                    
                    ed->detectEdges(img);ed->detectLines(new_edlines);
                }*/




                // ROS_WARN("add %d lines", newlsd.size());
                //加到kp2里面

                // {
                //     vector<int> lineLength;
                //     for (auto line : new_edlines)
                //         lineLength.push_back(sqrt((line[0] - line[2]) * (line[0] - line[2]) + (line[1] - line[3]) * (line[1] - line[3])));
                //     sort(lineLength.begin(), lineLength.end());
                //     if (!lineLength.empty())
                //         // LINELENGTH = lineLength[max(int(lineLength.size() * ratio), (int)lineLength.size() - 40)];
                //         LINELENGTH = lineLength[int(lineLength.size() * ratio)];

                //     else
                //         LINELENGTH = 0;
                //     // ROS_WARN("line Length: %f", LINELENGTH);
                // }
                int num = linenum - forw_img->vecLine.size() + addLineNum;
                int lineIndex = forw_img->vecLine.size();
                for (auto line : new_edlines)
                {
                    Line l(line);
                    bool ifMerge = false;
#if _CONTROLLINEDENSE_
                    // if (merge.at<uchar>(l.MidPt) != 255 &&
                    //     (ifMerge = mergeTrackedLine(l, merge, forw_img->vecLine, mergel)))
                    //     ;
                    if ((mergel.at<uchar>(l.MidPt) != 255 ||
                         mergel.at<uchar>(l.StartPt) != 255 ||
                         mergel.at<uchar>(l.EndPt) != 255) &&
                        (ifMerge = mergeLine(l, mergel, forw_img->vecLine, maskl)))
                        ;
                    else if ((mask.at<uchar>(l.MidPt) != 255) ||
                             (maskl.at<uchar>(l.MidPt) != 255) ||
                             num <= 0)
                        continue;

                    if (ifMerge)
                        continue;
#endif
                    // ROS_WARN("(%d,%d)mask", mask.at<uchar>((int)l.StartPt.x, (int)l.StartPt.y), mask.at<uchar>((int)l.EndPt.x, (int)l.EndPt.y));
                    // if (l.length >= LINELENGTH && goodLine(img, line))
                    if (l.length >= LINELENGTH)
                    {
                        forw_img->vecLine.push_back(l);
                        forw_img->success.push_back(-1);
                        forw_img->lineID.push_back(allfeature_cnt++);

#if _SHOWLINEFLOW_
                        LineST linest;
                        linest.start = l.StartPt;
                        linest.end = l.EndPt;

                        LineRecord linerecord;
                        linerecord.push_back(linest);

                        forw_img->lineRec.push_back(linerecord);

#endif
#if _ONELINE_
                        break;
#endif
                        num--;
                        // if (num <= 0)
                        //     break;
#if _CONTROLLINEDENSE_
                        cv::line(maskl, l.StartPt, l.EndPt, 0, 30);
                        drawLine(l, mergel, lineIndex++);
#endif
                    }
                }
                // imshow("maskl",maskl);
                // imshow("mergel", mergel);
                // waitKey();
            }
            double linedetect_time = t_linedetect.toc() ;
            ofstream foutD("/home/jiangdi/result_output/time/euroc/EPLF_VINS_WS/eplfvins_line_detect.csv", ofstream::app);
            foutD << linedetect_time<<endl;
            foutD.close();
            ofstream foutE("/home/jiangdi/result_output/time/euroc/EPLF_VINS_WS/eplfins_line_detect_tracker.csv", ios::app);
            foutE << lineflowtime+linedetect_time<< endl;
            foutE.close();
        }
    }
    cur_img = forw_img;
#endif
}
