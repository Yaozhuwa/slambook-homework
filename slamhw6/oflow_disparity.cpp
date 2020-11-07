#include <opencv2/opencv.hpp>
#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

using namespace std;
using namespace cv;

string LeftImage = "../left.png";
string RightImage = "../right.png";
string DispImage = "../disparity.png";

inline float GetPixelValue(const cv::Mat &img, float x, float y)
{
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]);
}

void OpticalFlowSingleLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false);

void OpticalFlowMultiLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false);

int main()
{
    Mat left = imread(LeftImage, IMREAD_GRAYSCALE);
    Mat right = imread(RightImage, IMREAD_GRAYSCALE);

    // Camera intrinsics
    // 内参
    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    // 基线
    double baseline = 0.573;

    // key points, using GFTT here.
    vector<KeyPoint> kp1;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
    detector->detect(left, kp1);

    // then test multi-level LK
    vector<KeyPoint> kp2_multi;
    vector<bool> success_multi;
    OpticalFlowMultiLevel(left, right, kp1, kp2_multi, success_multi, true);

    Mat imgLeft_show;
    Mat disparity = imread(DispImage, IMREAD_GRAYSCALE);
    cvtColor(left, imgLeft_show, CV_GRAY2BGR);

    int error=0;
    int cnt=0;
    for (int i = 0; i < kp2_multi.size(); i++)
    {
        if (success_multi[i])
        {
            cv::circle(imgLeft_show, kp1[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(imgLeft_show, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
            int disp = kp1[i].pt.x-kp2_multi[i].pt.x;
            if (disp<0) disp = -disp;
            uchar disp_truth = disparity.at<uchar>(kp1[i].pt.y, kp1[i].pt.x);
            error += disp>int(disp_truth)?disp-int(disp_truth):int(disp_truth)-disp;
            cnt++;
        }
    }
    cout<<"Disparity error average: "<<double(error)/cnt<<endl;

    cv::imshow("tracked multi level", imgLeft_show);
    cv::waitKey(0);
    return 0;
}

void OpticalFlowSingleLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse)
{

    // parameters
    int half_patch_size = 4;
    int iterations = 10;
    bool have_initial = !kp2.empty();

    for (size_t i = 0; i < kp1.size(); i++)
    {
        auto kp = kp1[i];
        double dx = 0, dy = 0; // dx,dy need to be estimated
        if (have_initial)
        {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded

        // Gauss-Newton iterations
        for (int iter = 0; iter < iterations; iter++)
        {
            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
            Eigen::Vector2d b = Eigen::Vector2d::Zero();
            cost = 0;

            if (kp.pt.x + dx <= half_patch_size || kp.pt.x + dx >= img1.cols - half_patch_size ||
                kp.pt.y + dy <= half_patch_size || kp.pt.y + dy >= img1.rows - half_patch_size)
            { // go outside
                succ = false;
                break;
            }

            // compute cost and jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++)
                {

                    // TODO START YOUR CODE HERE (~8 lines)
                    double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) - GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);
                    Eigen::Vector2d J; // Jacobian
                    if (inverse == false)
                    {
                        // Forward Jacobian
                        J[0] = -0.5 * (GetPixelValue(img2, kp.pt.x + x + dx + 1, kp.pt.y + y + dy) - GetPixelValue(img2, kp.pt.x + x + dx - 1, kp.pt.y + y + dy));
                        J[1] = -0.5 * (GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy + 1) - GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy - 1));
                    }
                    else
                    {
                        // Inverse Jacobian
                        // NOTE this J does not change when dx, dy is updated, so we can store it and only compute error
                        J[0] = -0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) - GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y));
                        J[1] = -0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) - GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1));
                    }

                    // compute H, b and set cost;
                    if (inverse == false || iter == 0)
                        H += J * J.transpose();
                    b += -J * error;
                    cost += error * error;
                    // TODO END YOUR CODE HERE
                }

            // compute update
            // TODO START YOUR CODE HERE (~1 lines)
            Eigen::Vector2d update;
            update = H.ldlt().solve(b);
            // TODO END YOUR CODE HERE

            if (isnan(update[0]) || isnan(update[1]))
            {
                // sometimes occurred when we have a black or white patch and H is irreversible
                cout << "update is nan" << endl;
                succ = false;
                break;
            }
            if (iter > 0 && cost > lastCost)
            {
                // cout << "cost increased: " << cost << ", " << lastCost << endl;
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;
        }

        success.push_back(succ);

        // set kp2
        if (have_initial)
        {
            kp2[i].pt = kp.pt + Point2f(dx, dy);
        }
        else
        {
            KeyPoint tracked = kp;
            tracked.pt += cv::Point2f(dx, dy);
            kp2.push_back(tracked);
        }
    }
}

void OpticalFlowMultiLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse)
{

    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    vector<Mat> pyr1, pyr2; // image pyramids
    // TODO START YOUR CODE HERE (~8 lines)
    for (int i = 0; i < pyramids; i++)
    {
        if (i == 0)
        {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
            continue;
        }
        Mat dst1, dst2;
        resize(img1, dst1, Size(), scales[i], scales[i]);
        resize(img2, dst2, Size(), scales[i], scales[i]);
        pyr1.push_back(dst1);
        pyr2.push_back(dst2);
    }
    // TODO END YOUR CODE HERE

    // coarse-to-fine LK tracking in pyramids
    // TODO START YOUR CODE HERE
    vector<KeyPoint> kp1_pyr, kp2_pyr;
    for (int level = pyramids - 1; level >= 0; level--)
    {
        kp1_pyr.clear();
        for (auto kp : kp1)
        {
            KeyPoint p;
            p.pt = kp.pt * scales[level];
            kp1_pyr.push_back(p);
        }

        if (level != pyramids - 1)
        {
            for (auto &kp : kp2_pyr)
            {
                kp.pt /= pyramid_scale;
            }
        }
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse);
    }
    for (auto &kp : kp2_pyr)
        kp2.push_back(kp);
    // TODO END YOUR CODE HERE
    // don't forget to set the results into kp2
}
