#include "parameters.h"

std::string IMAGE_TOPIC;
std::string IMU_TOPIC;
std::vector<std::string> CAM_NAMES;
std::string FISHEYE_MASK;
int MAX_CNT;
int MIN_DIST;
int WINDOW_SIZE;
int FREQ;
double F_THRESHOLD;
int SHOW_TRACK;
int STEREO_TRACK;
int EQUALIZE;
int ENABLE_REGION_PHOTOMETRIC;
int ENABLE_HUBER_PHOTOMETRIC;
int ROW;
int COL;
int FOCAL_LENGTH;
int FISHEYE;
bool PUB_THIS_FRAME;
Eigen::Matrix3d RCI;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(ros::NodeHandle &n)
{
    std::string config_file;
    config_file = readParam<std::string>(n, "config_file");
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings:" << config_file << std::endl;
    }
    std::string VINS_FOLDER_PATH = readParam<std::string>(n, "vins_folder");

    fsSettings["image_topic"] >> IMAGE_TOPIC;
    fsSettings["imu_topic"] >> IMU_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    FREQ = fsSettings["freq"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
EQUALIZE = fsSettings["equalize"];

// 兼容旧配置：YAML中没有字段时，默认开启当前算法。
ENABLE_REGION_PHOTOMETRIC = 1;
ENABLE_HUBER_PHOTOMETRIC = 1;

const cv::FileNode region_photometric_node =
    fsSettings["enable_region_photometric"];

if (!region_photometric_node.empty())
{
    region_photometric_node >>
        ENABLE_REGION_PHOTOMETRIC;
}

const cv::FileNode huber_photometric_node =
    fsSettings["enable_huber_photometric"];

if (!huber_photometric_node.empty())
{
    huber_photometric_node >>
        ENABLE_HUBER_PHOTOMETRIC;
}

ROS_INFO_STREAM(
    "Enable region photometric: "
    << ENABLE_REGION_PHOTOMETRIC);

ROS_INFO_STREAM(
    "Enable Huber photometric: "
    << ENABLE_HUBER_PHOTOMETRIC);

FISHEYE = fsSettings["fisheye"];
    if (FISHEYE == 1)
        FISHEYE_MASK = VINS_FOLDER_PATH + "config/fisheye_mask.jpg";
    CAM_NAMES.push_back(config_file);

    WINDOW_SIZE = 20;
    STEREO_TRACK = false;
    FOCAL_LENGTH = 460;
    PUB_THIS_FRAME = false;

    if (FREQ == 0)
        FREQ = 100;

    cv::Mat cv_R;
    fsSettings["extrinsicRotation"] >> cv_R;
    Eigen::Matrix3d eigen_R;
    cv::cv2eigen(cv_R, eigen_R);
    Eigen::Quaterniond Q(eigen_R);
    eigen_R = Q.normalized();
    RCI = eigen_R.transpose();
    ROS_INFO_STREAM("Extrinsic_R : " << std::endl
                                     << RCI);

    fsSettings.release();
}
