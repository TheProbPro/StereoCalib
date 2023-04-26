#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void stereoCalibration()
{
    //Load in calibration images (paths need to be correct!) (RGB is on the right):
    std::vector<cv::String> fileNames_left, fileNames_right;
    cv::glob("../../../TestIntrin/right/", fileNames_right, false);
    cv::glob("../../../TestIntrin/left/", fileNames_left, false);
    //Initialise size of chessboard (needs to be correct!):
    cv::Size BoardSize{ 6, 4 };
    std::vector<std::vector<cv::Point2f>> imagePoints_left, imagePoints_right;
    std::vector<std::vector<cv::Point3f>> object_points;
    //create vectors to contain corners
    std::vector<cv::Point2f> corners_left, corners_right;

    cv::Mat img_left, img_right, gray_left, gray_right;
    try {
        if (fileNames_left.size() != fileNames_right.size()) {
            throw std::runtime_error("Vectors are not of the same size.");
        }
        if (fileNames_left.empty() || fileNames_right.empty() || fileNames_left.back() == "" || fileNames_right.back() == "") {
            throw std::runtime_error("Last element in one or both vectors is not filled.");
        }
    }
    catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return;
    }

    // Detect feature points
    for (int i = 0; i < fileNames_left.size(); ++i) {
        // 1. Read in the image an call cv::findChessboardCorners()
        img_left = cv::imread(fileNames_left[i]);
        std::cout << "left" << img_left.type() << std::endl;
        img_right = cv::imread(fileNames_right[i]);
        std::cout << "right" << img_right.type() << std::endl;
        //std::cout << "RGB res: " << img_right.size() << " " << img_right.type() <<  " Tof res: " << img_left.size() << " " << img_left.type() << std::endl;
        cv::cvtColor(img_left, gray_left, cv::COLOR_BGR2GRAY);
        cv::cvtColor(img_right, gray_right, cv::COLOR_BGR2GRAY);

        bool found_left = cv::findChessboardCorners(img_left, BoardSize, corners_left, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);
        bool found_right = cv::findChessboardCorners(img_right, BoardSize, corners_right, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);

        // 2. Use cv::cornerSubPix() to refine the found corner detections
        if (!found_left || !found_right) {
            std::cout << "Chessboard find error!" << std::endl;
            std::cout << "leftImg: " << img_left << " and rightImg: " << img_right << std::endl;
            continue;
        }
        if (found_left) {
            cv::cornerSubPix(gray_left, corners_left, cv::Size(5, 5), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
            cv::drawChessboardCorners(img_left, BoardSize, corners_left, found_left);
        }
        if (found_right) {
            cv::cornerSubPix(gray_right, corners_right, cv::Size(5, 5), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
            cv::drawChessboardCorners(img_right, BoardSize, corners_right, found_right);
        }

        // Display (Can be removed after later but is good for quality ensurance)
        //cv::imshow("chessboard detection left", img_left);
        //cv::imshow("chessboard detection right", img_right);
        //cv::waitKey(0);

        // 3. Generate checkerboard (world) coordinates Q. The board has 7 x 5
       // fields with a size of 30x30mm
        int square_size = 30;
        std::vector<cv::Point3f> objp;
        for (int i = 0; i < BoardSize.height; i++) {
            for (int j = 0; j < BoardSize.width; j++) {
                objp.push_back(cv::Point3f(j * square_size, i * square_size, 0));
            }
        }
        
        if (found_left && found_right) {
            std::cout << i << ". Found Corners!" << std::endl;
            imagePoints_left.push_back(corners_left);
            imagePoints_right.push_back(corners_right);
            object_points.push_back(objp);
        }
    }

    std::vector<std::vector<cv::Point2f>> left_img_points, right_img_points;
    for (int i = 0; i < imagePoints_left.size(); i++) {
        std::vector<cv::Point2f> v1, v2;
        for (int j = 0; j < imagePoints_left[i].size(); j++) {
            v1.push_back(cv::Point2f((double)imagePoints_left[i][j].x, (double)imagePoints_left[i][j].y));
            v2.push_back(cv::Point2f((double)imagePoints_right[i][j].x, (double)imagePoints_right[i][j].y));
        }
        left_img_points.push_back(v1);
        right_img_points.push_back(v2);
    }

    // Load in the camera matrixeces and the distortion coefficients
    cv::Mat tmpr, tmpl;
    cv::Vec<float, 5> k_left, k_right;
    cv::FileStorage fileKr("../../../TestIntrin/Config/RGBIntrinsic.xml", cv::FileStorage::READ);
    fileKr["K"] >> tmpr;
    fileKr.release();
    cv::FileStorage filekr("../../../TestIntrin/Config/RGBDistortion.xml", cv::FileStorage::READ);
    filekr["k"] >> k_right;
    filekr.release();
    cv::FileStorage fileKl("../../../TestIntrin/Config/ToFIntrinsic.xml", cv::FileStorage::READ);
    fileKl["K"] >> tmpl;
    fileKl.release();
    cv::FileStorage filekl("../../../TestIntrin/Config/ToFDistortion.xml", cv::FileStorage::READ);
    filekl["k"] >> k_left;
    filekl.release();
    //cv::Matx33f newK_right(tmpr);
    //cv::Matx33f newK_left(tmpl);

    cv::Mat newK_right = cv::getOptimalNewCameraMatrix(tmpr, k_right, img_right.size(), 1, img_right.size());
    cv::Mat newK_left = cv::getOptimalNewCameraMatrix(tmpl, k_left, img_left.size(), 1, img_left.size());


    //Stereo calibarate
    cv::Mat R, F, E;
    cv::Vec3d T;
    int flag = 0;
    flag |= cv::CALIB_FIX_INTRINSIC + cv::CALIB_USE_INTRINSIC_GUESS;
    std::cout << "Read Intrinsics" << std::endl;
    //cv::TermCriteria criteria_stereo = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);
    cv::stereoCalibrate(object_points, left_img_points, right_img_points, newK_left, k_left, newK_right, k_right, img_left.size(), R, T, E, F);
    std::cout << "Done Calibration!" << std::endl;
    std::cout << "Starting Rectification" << std::endl;
    //stereo rectify
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(newK_left, k_left, newK_right, k_right, img_right.size(), R, T, R1, R2, P1, P2, Q);

    std::cout << "R: " << R << "\n T: " << T << "\n R1: " << R1 << "\n R2: " << R2 << "\n P1: " << P1 << "\n P2: " << P2 << "\n Q: " << Q << std::endl;

    //Create maps to rectify images with
    cv::Mat lmapx, lmapy, rmapx, rmapy;
    cv::Mat imgU1, imgU2;
    img_left = cv::imread(fileNames_left[15]);
    img_right = cv::imread(fileNames_right[15]);
    cv::initUndistortRectifyMap(newK_left, k_left, R1, P1, img_left.size(), CV_32F, lmapx, lmapy);
    cv::initUndistortRectifyMap(newK_right, k_right, R2, P2, img_right.size(), CV_32F, rmapx, rmapy);
    cv::remap(img_left, imgU1, lmapx, lmapy, cv::INTER_LINEAR);
    cv::remap(img_right, imgU2, rmapx, rmapy, cv::INTER_LINEAR);
    std::cout << imgU1.size() << std::endl;
    cv::Mat concat;
    cv::hconcat(imgU1, imgU2, concat);
    //Show undistorted images
    cv::imshow("rectified left", imgU1);
    cv::imshow("rectified right", imgU2);
    cv::imshow("Concat", concat);
    cv::waitKey(0);

    cv::FileStorage fileRGB("../../../TestIntrin/Config/RGBStereo.xml", cv::FileStorage::WRITE);
    fileRGB << "mapx" << rmapx;
    fileRGB << "mapy" << rmapy;
    cv::FileStorage fileToF("../../../TestIntrin/Config/ToFStereo.xml", cv::FileStorage::WRITE);
    fileToF << "mapx" << lmapx;
    fileToF << "mapy" << lmapy;
}


int main(int argc, char** argv) {

    stereoCalibration();
     // Show image calibration
    /*cv::Mat img_left = cv::imread("../../../TestIntrin/left/00000000", cv::IMREAD_GRAYSCALE);

    cv::Mat tmpl;
    cv::Vec<float, 5> k_left;
    cv::FileStorage fileKl("../../../TestIntrin/Config/ToFIntrinsic.xml", cv::FileStorage::READ);
    fileKl["K"] >> tmpl;
    fileKl.release();
    cv::FileStorage filekl("../../../TestIntrin/Config/ToFDistortion.xml", cv::FileStorage::READ);
    filekl["k"] >> k_left;
    filekl.release();
    cv::Matx33f K_left(tmpl);

    cv::Mat mapX, mapY, out;
    cv::initUndistortRectifyMap(K_left, k_left, cv::Matx33f::eye(), K_left, img_left.size(), CV_32FC1, mapX, mapY);
    cv::remap(img_left, out, mapX, mapY, cv::INTER_LINEAR);
    cv::imshow("Test", out);
    cv::waitKey(0);
    */
    return 0;
}