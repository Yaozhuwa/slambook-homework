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

string TrajectoryFile = "../compare.txt";
void ICP(vector<Vector3d> pts_est, vector<Vector3d> pts_gt, Matrix3d &R, Vector3d &t);
void DrawTrajectory(vector<Sophus::SE3d> poses1, vector<Sophus::SE3d> poses2, string winName="Trajectory Viewer");

int main(){
    ifstream fin(TrajectoryFile);
    vector<Vector3d> pts_est,pts_gt;
    vector<Sophus::SE3d> poses_gt, poses_est;
    while(!fin.eof()){
        double time, tx,ty,tz,qx,qy,qz,qw;
        fin>>time>>tx>>ty>>tz>>qx>>qy>>qz>>qw;
        Quaterniond q(qw,qx,qy,qz);
        Vector3d t_est(tx,ty,tz);
        Sophus::SE3d SE3_est(q,t_est);
        pts_est.push_back(t_est);
        poses_est.push_back(SE3_est);

        fin>>time>>tx>>ty>>tz>>qx>>qy>>qz>>qw;
        Quaterniond q2(qw,qx,qy,qz);
        Vector3d t_gt(tx,ty,tz);
        Sophus::SE3d SE3_gt(q2,t_gt);
        poses_gt.push_back(SE3_gt);
        pts_gt.push_back(t_gt);
    }
    Matrix3d R;
    Vector3d t;
    ICP(pts_est, pts_gt, R, t);
    Sophus::SE3d SE3_E2G(R, t);

    int N = poses_gt.size();
    vector<Sophus::SE3d> poses_after(N);
    for(int i=0;i<N;i++){
        poses_after[i] = SE3_E2G*poses_est[i];
    }
    DrawTrajectory(poses_gt, poses_est, "before");
    DrawTrajectory(poses_gt, poses_after, "after");

    double error=0;
    for (int i=0;i<N;i++){
        Sophus::SE3d SE3_differ = poses_gt[i].inverse()*poses_after[i];
        typedef Eigen::Matrix<double, 6, 1> Vector6d;
        Vector6d se3_differ = SE3_differ.log();
        error += se3_differ.dot(se3_differ);
    }
    double RMSE = sqrt(error/N);
    cout<<"RMSE: "<<RMSE<<endl;

    return 0;
}

void ICP(vector<Vector3d> pts_est, vector<Vector3d> pts_gt, Matrix3d &R, Vector3d &t){
    Vector3d pe(0,0,0), pg(0,0,0);
    int N = pts_est.size();
    for(int i=0;i<N;i++){
        pe += pts_est[i];
        pg += pts_gt[i];
    }
    pe /= N;
    pg /= N;
    vector<Vector3d> q_est(N), q_gt(N);
    for (int i=0;i<N;i++){
        q_est[i] = pts_est[i]-pe;
        q_gt[i] = pts_gt[i]-pg;
    }
    Matrix3d W = Matrix3d::Zero();
    for (int i=0;i<N;++i){
        W += q_gt[i]*q_est[i].transpose();
    }
    cout<<"W="<<W<<endl;
    JacobiSVD<Matrix3d> svd(W, ComputeFullU|ComputeFullV);
    Matrix3d U = svd.matrixU();
    Matrix3d V = svd.matrixV();
    cout<<"U="<<U<<endl;
    cout<<"V="<<V<<endl;
    Matrix3d R_ = U*(V.transpose());
    if(R_.determinant()<0){
        R_ = -R_;
    }
    
    Vector3d t_ = pg-R_*pe;
    R = R_;
    t = t_;
}

void DrawTrajectory(vector<Sophus::SE3d> poses1, vector<Sophus::SE3d> poses2, string winName) {

    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind(winName, 1024, 768);
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