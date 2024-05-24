#include <iostream>
#include <chrono>
#define EIGEN_NO_DEBUG
//#define EIGEN_USE_MKL_ALL
#include <eigen3/Eigen/Dense>
#include <omp.h>

using namespace std;
using namespace Eigen;
using namespace chrono;

void Dg(MatrixXd a, MatrixXd b, MatrixXd c, MatrixXd d);
void Estimation(MatrixXd a, MatrixXd b, MatrixXd c, MatrixXd d, VectorXd Xsio);
void Define();

const int dim=2,DelP = 100, DelE = 250, Tf=1000, n=Tf+DelP+1;
const double T = 0.001;
int j, d, b;




MatrixXd phf(2, 2), prf(2, 2), Qr(2, 2), Qh(2, 2), mA(2, 2), Brt_i, Bht_i, Brht_i, Art_i, Aht_i, crt_i, cht_i, Frt_i, Rr(1, 1), Rh(1, 1), Rrh(1, 1), Fht_i, Pr(2, 2), Ph(2, 2), a, br, bh, c;
VectorXd ahf(2), arf(2), mBr(2), mBh(2), mC(2), Ur(1), Uh(1), Xsi0(2), Xsi(2), error(2), Uh_arr[DelE], Xsi_arr[DelE], ar(2), ah(2);
MatrixXd A(dim * n, dim), Br(dim * n, 1), Bh(dim * n, 1), C(dim * n, 1);

int main() {
    Define();
    for (int i=0; i<n; i++) {
        j = 2 * i;
        A.block(j, 0, 2, 2) = mA;
        Br.block(j, 0, 2, 1) = mBr;
        Bh.block(j, 0, 2, 1) = mBh;
        C.block(j, 0, 2, 1) = mC;
    }

    // Start clock
    auto start = high_resolution_clock::now();

    for (int i=0; i<Tf; i++) {      j=2*i;b=(j+DelP*2)-1;
        Dg(mA, mBr, mBh, mC);

        // Updating Ur, Uh, and Xsi
        //Ur = -Rr.inverse() * mBr.transpose() * (Pr * Xsi + ar);
        //Uh = -Rh.inverse() * mBh.transpose() * (Ph * Xsi + ah);
        //Xsi += T * (mA* Xsi + mBr* Ur + mBh * Uh + mC);

        // Store Uh and Xsi in the array
        //Uh_arr[i % DelE] = Uh;
        //Xsi0 = Xsi_arr[i % DelE];
        //Xsi_arr[i % DelE] = Xsi;

        //if (i>DelE) {     j-=(DelE/2)-2;
            //Estimation(A.block(j, 0, b-j+1, 2), Br.block(j, 0, b-j+1, 1), Bh.block(j, 0, b-j+1, 1), C.block(j, 0, b-j+1, 1), Xsi0);
        //}

    }


    // End clock
    auto end = high_resolution_clock::now();


    auto duration = duration_cast<microseconds>((end - start)/Tf);

    // Print 
    // cout << "Pr:\n" << Pr << endl;
    // cout << "ar:\n" << ar << endl;
    // cout << "Ph:\n" << Ph << endl;
    // cout << "ah:\n" << ah << endl;

    // cout << "Ur:\n" << Ur << endl;
    // cout << "Uh:\n" << Uh << endl;
    // cout << "Xsi:\n" << Xsi << endl;

    cout << "Time taken: " << duration.count() << " microseconds" << endl;    // Print Time

    return 0;
}


void Define(){
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
    Xsi0 << 0, 0;
    Xsi << 1, 1;
    error << 0, 0;
    Rr << 1;
    Rh << 1;
    Rrh << 1;
}

void Dg(MatrixXd a, MatrixXd br, MatrixXd bh, MatrixXd c) {
    // Initialize loop variables with actual values
    Pr = prf;
    Ph = phf;
    ar = arf;
    ah = ahf;

    // Implement the loop
    for (int i=0; i<DelP; ++i) {    d=2*i;
        //a=dgA.block(d, 0, 2, 2); br=dgBr.block(d, 0, 2, 1); bh=dgBh.block(d, 0, 2, 1); c=dgC.block(d, 0, 2, 1);

        Brt_i = br * Rr.inverse() * br.transpose();
        Bht_i = bh * Rh.inverse() * bh.transpose();
        Brht_i = br * Rr.inverse() * Rrh * Rr.inverse() * br.transpose();

        Art_i = a - Bht_i * Ph;
        Aht_i = a - Brt_i * Pr;
        crt_i = c - Bht_i * ah;
        cht_i = c - Brt_i * ar;

        Frt_i = Pr * Art_i;
        Fht_i = Ph * Aht_i;

        // Update equations
        Pr = Pr + T * (Frt_i + Frt_i.transpose() + Qr - Pr * Brt_i * Pr + Ph * Brht_i * Ph);
        ar = ar + T * ((Art_i - Brt_i * Pr).transpose() * ar + Pr * crt_i + Ph * Brht_i * ah);
        Ph = Ph + T * (Fht_i + Fht_i.transpose() + Qh - Ph * Bht_i * Ph);
        ah = ah + T * ((Aht_i - Bht_i * Ph).transpose() * ah + Ph * cht_i);

    }
}

void Estimation(MatrixXd eA, MatrixXd eBr, MatrixXd eBh, MatrixXd eC, VectorXd Xsio) {
    Xsi = Xsio;
    error.setZero();

    for (int i = 0; i < DelE; i++) {    d=2*i;b=(d+DelP*2)-1;

        Dg(eA.block(d, 0, b-d+1, 2), eBr.block(d, 0, b-d+1, 1), eBh.block(d, 0, b-d+1, 1), eC.block(d, 0,b-d+1, 1));

        Uh = -Rh.inverse() * Bh.block(d, 0, 2, 1).transpose() * (Ph * Xsi + ah);
        Ur = -Rr.inverse() * Br.block(d, 0, 2, 1).transpose() * (Pr * Xsi + ar);
        Xsi += T * (A.block(d, 0, 2, 2) * Xsi + Br.block(d, 0, 2, 1) * Ur + Bh.block(d, 0, 2, 1) * Uh + C.block(d, 0, 2, 1));

        error += (Uh_arr[i % DelE] - Uh).cwiseAbs();
    }

    cout << "Error:\n" << error << endl;
}

