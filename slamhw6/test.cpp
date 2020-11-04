#include <iostream>
#include <vector>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <Eigen/Core>

using namespace Eigen;
using namespace std;

int main(){
    int a=0;
    vector<int> b;
    b.push_back(a);
    b[0]=1;
    cout<<"Test:"<<endl;
    cout<<a<<endl;
    cout<<b[0]<<endl;
    for (auto &i:b){
        i+=3;
    }
    cout<<b[0]<<endl;
    Sophus::SE3d SE33;
    Sophus::SO3d SO33;
    cout<<SE33.matrix()<<endl;
    std::cout<<SO33.matrix()<<endl;
    
    return 0;
}