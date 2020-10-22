#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>

using namespace std;
using namespace Eigen;

int main(){
    const int n = 100;
    MatrixXd a = MatrixXd::Random(n,n);
    MatrixXd A;
    A.noalias() = a * a.transpose();
    VectorXd b = VectorXd::Random(n);
    VectorXd x;
    
    cout<<"直接求逆"<<endl;
    auto t0 = chrono::steady_clock::now();
    x = A.inverse()*b;
    auto t1 = chrono::steady_clock::now();
    cout<<"Cost time: "<< chrono::duration<double,std::milli>(t1-t0).count()<<"ms"<<endl;
    cout<<"x= "<<x.transpose()<<endl;

    cout<<"QR 分解"<<endl;
    auto t2 = chrono::steady_clock::now();
    x = A.colPivHouseholderQr().solve(b);
    auto t3 = chrono::steady_clock::now();
    cout<<"Cost time: "<< chrono::duration<double,std::milli>(t3-t2).count()<<"ms"<<endl;
    cout<<"x= "<<x.transpose()<<endl;

    cout<<"Cholesky 分解"<<endl;
    auto t4 = chrono::steady_clock::now();
    x = A.ldlt().solve(b);
    auto t5 = chrono::steady_clock::now();
    cout<<"Cost time: "<< chrono::duration<double,std::milli>(t5-t4).count()<<"ms"<<endl;
    cout<<"x= "<<x.transpose()<<endl;

    return 0;
}