#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>

#ifdef __ANDROID__
#include <android/log.h>
#endif

using namespace cv;
using namespace std;

long long int get_now() {
    return chrono::duration_cast<std::chrono::milliseconds>(
            chrono::system_clock::now().time_since_epoch()
    ).count();
}

void platform_log(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
#ifdef __ANDROID__
    __android_log_vprint(ANDROID_LOG_VERBOSE, "ndk", fmt, args);
#else
    vprintf(fmt, args);
#endif
    va_end(args);
}

// Avoiding name mangling
extern "C" {
    // Attributes to prevent 'unused' function from being removed and to make it visible
    __attribute__((visibility("default"))) __attribute__((used))
    const char* version() {
        return CV_VERSION;
    }

    __attribute__((visibility("default"))) __attribute__((used))
    void process_image(char* inputImagePath, char* outputImagePath) {
        long long start = get_now();
        
        Mat input = imread(inputImagePath, IMREAD_GRAYSCALE);
        //Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
        Mat threshed, withContours;

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;

        adaptiveThreshold(input, threshed, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 77, 6);
        findContours(threshed, contours, hierarchy, RETR_TREE, CHAIN_APPROX_TC89_L1);

        cvtColor(threshed, withContours, COLOR_GRAY2BGR);
        drawContours(withContours, contours, -1, Scalar(0, 255, 0), 4);

        imwrite(outputImagePath, withContours);
        
        int evalInMillis = static_cast<int>(get_now() - start);
        platform_log("Processing done in %d ms\n", evalInMillis);
    }

    __attribute__((visibility("default"))) __attribute__((used))
        const char* create_rectified_image(char* inputImagePath, char* outputImagePath) {
            long long start = get_now();

            Mat input = imread(inputImagePath, IMREAD_GRAYSCALE);

            std::vector<int> markerIds;
            std::vector<std::vector<Point2f>> markerCorners, rejectedCandidates;
            Ptr<aruco::DetectorParameters> parameters = aruco::DetectorParameters::create();
            Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);

            aruco::detectMarkers(input, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

            platform_log("loaded image width: %d height: %d)", input.cols, input.rows);
            platform_log("num markers found: %d", markerIds.size());

            // alternative RANSAC approach: https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/

            Point2f corner_top_left;
            Point2f corner_top_right;
            Point2f corner_bottom_left;
            Point2f corner_bottom_right;
            enum CornerArucoCodes {topLeft=20, topRight=21, bottomLeft=22, bottomRight=23};

            int number_of_corner_markers_found = 0;

            for (int mIdx = 0; mIdx < markerIds.size(); ++mIdx) {
                const int markerId = markerIds[mIdx];
                Point2f center(0.f, 0.f);
                for(int i = 0; i < 4; ++i) {
                    center += markerCorners[mIdx][i];
                    platform_log("marker id %d corner %d: %fx %fy\n", markerId, i, markerCorners[mIdx][i].x, markerCorners[mIdx][i].y);
                }
                center /= 4.f;
                platform_log("marker id %d center: %f %f\n", markerId, center.x, center.y);

                switch (markerId) {
                    case CornerArucoCodes::topLeft:
                        number_of_corner_markers_found++;
                        corner_top_left = center;
                        break;
                    case CornerArucoCodes::topRight:
                        number_of_corner_markers_found++;
                        corner_top_right = center;
                        break;
                    case CornerArucoCodes::bottomLeft:
                        number_of_corner_markers_found++;
                        corner_bottom_left = center;
                        break;
                    case CornerArucoCodes::bottomRight:
                        number_of_corner_markers_found++;
                        corner_bottom_right = center;
                        break;
                }
            }

            bool found_all_markers = (number_of_corner_markers_found == 4);

            const Point2f detectedPoints[] = {corner_top_left, corner_top_right, corner_bottom_right, corner_bottom_left};
            const Point2f imageBoundaries[] = {Point2f(0.0, 0.0), Point2f(input.cols - 1, 0.0), Point2f(input.cols - 1, input.rows - 1), Point2f(0.0, input.rows - 1)};
            
            Mat perspective_transform_matrix = getPerspectiveTransform(detectedPoints, imageBoundaries);
            
            Mat rectifiedImage;

            warpPerspective(input, rectifiedImage, perspective_transform_matrix, input.size());

            imwrite(outputImagePath, rectifiedImage);

            //Mat outputImage = input.clone();
            //aruco::drawDetectedMarkers(outputImage, markerCorners, markerIds);
            //imwrite(outputImagePath, outputImage);

            int evalInMillis = static_cast<int>(get_now() - start);
            platform_log("Rectifying done in %d ms\n", evalInMillis);
            return CV_VERSION;
        }
}
