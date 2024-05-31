#include <iostream>
#include <chrono>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;
using namespace chrono;

struct Control;
struct Planning;
struct Estimation;

struct Control{
  const int dim=2,DelP = 100, DelE = 250, Tf=1000, n=Tf+DelP+1;
  const double Ts = 0.001;
  MatrixXd a, br, bh, c;

  Control() : a(n*dim, dim), br(n*dim, 1), bh(n*dim, 1), c(n*dim, 1) {  }

  void SetMatrices(){
    MatrixXd mA(2, 2);
    VectorXd mBr(2), mBh(2), mC(2);
    mA << 0, 1,  -0.1, -0.1;
    mBr << 0, 0.1;  mBh << 0, 0.1;  mC << 0, 0.1;
    for (int i=0; i<n; i++) { int j = 2 * i;
      a.block(j, 0, 2, 2) = mA; br.block(j, 0, 2, 1) = mBr; bh.block(j, 0, 2, 1) = mBh; c.block(j, 0, 2, 1) = mC;
    }
  }
};

struct Planning{
  MatrixXd phf, prf, Qr, Qh, Rr, Rh, Rrh;
  VectorXd ahf, arf;
  MatrixXd A, Br, Bh, C;

  Planning(const Control& c) : phf(c.dim, c.dim), prf(c.dim, c.dim), Qr(c.dim, c.dim), Qh(c.dim, c.dim), Rr(c.dim/2, c.dim/2), Rh(c.dim/2, c.dim/2), Rrh(c.dim/2, c.dim/2),
      ahf(c.dim), arf(c.dim), A(c.n*c.dim, c.dim), Br(c.n*c.dim, c.dim/2), Bh(c.n*c.dim, c.dim/2), C(c.n*c.dim, c.dim/2) {   }
  
  void SetMatrices(const Control& c){
    A=c.a;  Bh=c.bh;  Br=c.br;  C=c.c;
    phf << 0, 0, 0, 0;  prf = phf;
    ahf << 0, 0;    arf << 0, 0; 
    Qr << 10, 0,  0, 0.1;   Qh << 20, 0,  0, 0.1;  
    Rr << 1;    Rh << 1;    Rrh << 1;
  }
};

struct Estimation{
   VectorXd Ur, Uh, ξ0, ξ, error, Uh_arr[DelE], ξ_arr[DelE];
};




int main(){
  Control C;
  C.SetMatrices();

  Planning P(C);
  P.SetMatrices(C);

  return 0;
}

