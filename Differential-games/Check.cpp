#include <iostream>
#include <chrono>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;
using namespace chrono;

void Dg(MatrixXd a, MatrixXd b, MatrixXd c, MatrixXd d);
void Estimation(MatrixXd a, MatrixXd b, MatrixXd c, MatrixXd d, VectorXd ξo);
void Define();

const int dim = 2, DelP = 55, DelE = 5, Tf = 50, n = Tf + DelP + 1;
const double T = 0.001;
int j, d, b;

// Define initial values
MatrixXd phf(2, 2), prf(2, 2), Qr(2, 2), Qh(2, 2), mA(2, 2), Brt_i, Bht_i, Brht_i, Art_i, Aht_i, crt_i, cht_i, Frt_i, Rr(1, 1), Rh(1, 1), Rrh(1, 1), Fht_i, Pr(2, 2), Ph(2, 2), a, br, bh, c;
VectorXd ahf(2), arf(2), mBr(2), mBh(2), mC(2), Ur(1), Uh(1), ξ0(2), ξ(2), error(2), Uh_arr[DelE], ξ_arr[DelE], ar(2), ah(2);
MatrixXd A(dim * n, dim), Br(dim * n, 1), Bh(dim * n, 1), C(dim * n, 1);


int main(){

    A.setZero();
    Br.setZero();
    Bh.setZero();
    C.setZero();

    phf << 0, 0, 0, 0;
    prf = phf;
    Qr << 10, 0,
     0, 0.1;
    Qh << 20, 0,
     0, 0.1;
    mA << 0, 1, 
    -0.1, -0.1;
    ahf << 0, 0;
    arf << 0, 0;
    mBr << 0, 0.1;
    mBh << 0, 0.1;
    mC << 0, 0.1;
    Ur << 0;
    Uh << 0;
    ξ0 << 0, 0;
    ξ << 1, 1;
    error << 0, 0;
    Rr << 1;
    Rh << 1;
    Rrh << 1;
    Pr = MatrixXd::Zero(dim, dim);
    Ph = MatrixXd::Zero(dim, dim);


    return 0;
}