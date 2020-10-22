#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <vector>
#include <cmath>
// need pangolin for plotting trajectory
#include <pangolin/pangolin.h>
using namespace std;
using namespace Eigen;

string gtFile = "../groundtruth.txt";
string estFile = "../estimated.txt";

void DrawTrajectory(vector<Sophus::SE3d> poses1, vector<Sophus::SE3d> poses2) {

    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));


    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glLineWidth(2);
        for (size_t i = 0; i < poses1.size() - 1; i++) {
            glColor3f(1, 0.0f, 0);
            glBegin(GL_LINES);
            auto p1 = poses1[i], p2 = poses1[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        for (size_t i = 0; i < poses2.size() - 1; i++) {
            glColor3f(0, 0.0f, 1);
            glBegin(GL_LINES);
            auto p1 = poses2[i], p2 = poses2[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }

        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }

}

int main(){
    ifstream gtFin(gtFile);
    ifstream estFin(estFile);
    double error = 0;
    int count = 0;
    vector<Sophus::SE3d> poses_gt;
    vector<Sophus::SE3d> poses_est;
    while(!gtFin.eof() && !estFin.eof()){
        double time, tx,ty,tz,qx,qy,qz,qw;
        gtFin>>time>>tx>>ty>>tz>>qx>>qy>>qz>>qw;
        Quaterniond q(qw,qx,qy,qz);
        Vector3d t(tx,ty,tz);
        Sophus::SE3d SE3_gt(q,t);
        poses_gt.push_back(SE3_gt);
        estFin>>time>>tx>>ty>>tz>>qx>>qy>>qz>>qw;
        Quaterniond q2(qw,qx,qy,qz);
        Vector3d t2(tx,ty,tz);
        Sophus::SE3d SE3_est(q2,t2);
        poses_est.push_back(SE3_est);
        Sophus::SE3d SE3_differ = SE3_gt.inverse()*SE3_est;
        typedef Eigen::Matrix<double, 6, 1> Vector6d;
        Vector6d se3_differ = SE3_differ.log();
        error += se3_differ.dot(se3_differ);
        count++;
    }
    double RMSE = sqrt(error/count);
    cout<<"RMSE: "<<RMSE<<endl;
    DrawTrajectory(poses_gt,poses_est);
    // DrawTrajectory(poses_est);
    return 0;
}