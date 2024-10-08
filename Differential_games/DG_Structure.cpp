#include "DG_Structure.h"

#include <vector>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <sstream>
#include <thread>
#include <string>
#include <cmath>
#include <chrono>
#include <nlopt.hpp>
#include <iomanip>
#include <atomic>

using namespace std;
using namespace Eigen;
using namespace chrono;
using namespace nlopt;


//////////////////////////////////////////////////////////////////////////  Human   //////////////////////////////////////////////////////////////////////////

void Construct_Human(Dg::Human &H, const Dg::Constants &C) {
    H.qh.resize(2, 2);
    H.qh << 30, 0, 0, 0.01;
    H.rh.resize(1, 1);
    H.rh << 1;
    H.rhr.resize(1, 1);
    H.rhr << 0;
    H.Uh = VectorXd::Zero(C.m);
    H.Uh_arr.clear();
}

//////////////////////////////////////////////////////////////////////////  Robot   //////////////////////////////////////////////////////////////////////////

void Construct_Robot(Dg::Robot &R, const Dg::Constants &C) {
    R.qr.resize(2, 2);
    R.qr << 30, 0, 0, 3;
    R.rr.resize(1, 1);
    R.rr << 1;
    R.rrh.resize(1, 1);
    R.rrh << 0;
    R.Ur = VectorXd::Zero(C.m);
    R.Ur_arr.clear();
}

//////////////////////////////////////////////////////////////////////////  State   //////////////////////////////////////////////////////////////////////////

vector<string> split(const string &s, char delimiter, const Dg::Constants &C) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

void readCSV(const string &filename, Dg::State &S, const Dg::Constants &C) {
    ifstream file(filename);
    string line;
    int t = C.ntotal + C.np+1;

    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    MatrixXd mA(C.n, C.n), mB(C.n, C.m), mC(C.n, 1), mxd(C.n, 1);
    mB << 0, 1 / S.I;
    mA << 0, 1, 0, -S.D / S.I;
    mC << 0, 0;
    vector<string> parts;
    int i = 0;
    while (getline(file, line) && i < t) {
        parts = split(line, ',',C);

        mxd(0, 0)= stod(parts[0]);  mxd(1, 0)= stod(parts[1]); 
        mA(1, 0) = S.M * S.g * S.l * sin(stod(parts[0])) / S.I;
        mC(1, 0) = -(1 / S.I) * (S.I * stod(parts[2]) + S.D * stod(parts[1]) + S.M * S.g * S.l * cos(stod(parts[0])));

        S.A.block(0, C.n * i, C.n, C.n) = mA;
        S.B.block(0, i * C.m, C.n, C.m) = mB;
        S.C.block(0, i, C.n, 1) = mC;
        S.Xd.block(0,i, C.n, 1) =mxd;
        i++;
    }

    file.close();
}

void Construct_State(Dg::State &S, Dg::Human &H, Dg::Robot &R, const Dg::Constants &C) {
    S.M = H.m + R.m;
    S.l = ((H.m * H.m) + (R.m * R.l)) / S.M;
    S.I = H.I + R.I;
    S.D = R.D + H.D;
    S.xi = VectorXd::Ones(C.n);
    S.xi_arr.push_back(S.xi);
    string filename ="/home/ray/Downloads/trajectory.csv";
    S.A.resize(C.n, C.n * (C.ntotal +  C.np+1));
    S.B.resize(C.n, (C.ntotal +  C.np+1));
    S.C.resize(C.n, (C.ntotal +  C.np+1));
    S.Xd.resize(C.n, (C.ntotal +  C.np+1));

    readCSV(filename, S, C);
}

//////////////////////////////////////////////////////////////////////////  Planning //////////////////////////////////////////////////////////////////////////

void Construct_Planning(Dg::Planning &P, const Dg::State &S, Dg::Human &H, Dg::Robot &R, const Dg::Constants &C) {
    P.A.resize(C.n, C.n * C.np);
    P.Br.resize(C.n, C.np * C.m);
    P.Bh.resize(C.n, C.np * C.m);
    P.C.resize(C.n, C.np * C.m);
    P.A = S.A.block(0, 0, C.n, C.np * C.n);
    P.Br = S.B.block(0, 0, C.n, C.np * C.m);
    P.Bh = S.B.block(0, 0, C.n, C.np * C.m);
    P.C = S.C.block(0, 0, C.n, C.np);
    P.Pr.resize(C.n, C.n);
    P.Pr.setZero();
    P.Ph.resize(C.n, C.n);
    P.Ph.setZero();
    P.ar.resize(C.n);
    P.ar.setZero();
    P.ah.resize(C.n);
    P.ah.setZero();
    P.Rrh = H.rhr;
    P.Rh = H.rh;
    P.Rr = R.rr;
    P.Qh = H.qh;
    P.Qr = R.qr;
}

////////////////////////////////////////////////////////////////////////// Estimation //////////////////////////////////////////////////////////////////////////

void Construct_Estimation(Dg::Estimation &E, const Dg::State &S, Dg::Human &H, Dg::Robot &R, const Dg::Constants &C) {
    E.A.resize(C.n, C.n * (C.ne+ C.np));
    E.Br.resize(C.n, (C.ne+ C.np) * C.m);
    E.Bh.resize(C.n, (C.ne+ C.np) * C.m);
    E.C.resize(C.n, (C.ne+ C.np) * C.m);
    E.A = S.A.block(0, 0, C.n, (C.ne+ C.np) * C.n);
    E.Br = S.B.block(0, 0, C.n, (C.ne+ C.np) * C.m);
    E.Bh = S.B.block(0, 0, C.n, (C.ne+ C.np) * C.m);
    E.C = S.C.block(0, 0, C.n, (C.ne+ C.np));

    E.Pr.resize(C.n, C.n);
    E.Pr.setZero();
    E.Ph.resize(C.n, C.n);
    E.Ph.setZero();
    E.ar.resize(C.n);
    E.ar.setZero();
    E.ah.resize(C.n);
    E.ah.setZero();
    E.Rrh = H.rhr;
    E.Rh = H.rh;
    E.Rr = R.rr;
    E.Qh = H.qh;
    E.Qr = R.qr;

    E.Ur = R.Ur;
    E.Uh = H.Uh;
    E.xi = S.xi;
    E.Uh_arr.clear();
    E.X.clear();
}

////////////////////////////////////////////////////////////////////////// Update Planning //////////////////////////////////////////////////////////////////////////

void Update_Planning(Dg::Planning &P, const Dg::State &S, const Dg::Constants &C) {
    P.A = S.A.block(0, C.nc*C.n, C.n, C.np*C.n);
    P.Br= S.B.block(0,C.nc*C.m, C.n, C.np*C.m);
    P.Bh= S.B.block(0,C.nc*C.m, C.n, C.np*C.m);
    P.C = S.C.block(0, C.nc, C.n, C.np);
    
    P.ah.setZero(); P.ar.setZero(); P.Ph.setZero(); P.Pr.setZero();
}

////////////////////////////////////////////////////////////////////////// DG_fxn //////////////////////////////////////////////////////////////////////////

void DG_fxn(Dg::Planning &P, const Dg::Constants &C) {
    MatrixXd Brt_i,Brht_i,Bht_i,cht_i,crt_i;
    MatrixXd Aht_i,Art_i,Frt_i,Fht_i;
    MatrixXd Rr_inv = P.Rr.inverse(),Rh_inv = P.Rh.inverse();
    int j;

    for (int j = C.np-1; j > 0; j-=C.integration_step) {  

        Brt_i.noalias() = P.Br.block(0,j,C.n,C.m) * Rr_inv * (P.Br.block(0,j,C.n,C.m)).transpose();
        Bht_i.noalias() = P.Bh.block(0,j,C.n,C.m) * Rh_inv * (P.Bh.block(0,j,C.n,C.m)).transpose();
        Brht_i.noalias() = P.Br.block(0,j,C.n,C.m) * Rr_inv * P.Rrh * Rr_inv * (P.Br.block(0,j,C.n,C.m)).transpose();
        Art_i.noalias() = P.A.block(0,j*C.n,C.n,C.n) - Bht_i * P.Ph;
        Aht_i.noalias() = P.A.block(0,j*C.n,C.n,C.n) - Brt_i * P.Pr;
        crt_i.noalias() = P.C.block(0,j,C.n,C.m) - Bht_i * P.ah;
        cht_i.noalias() = P.C.block(0,j,C.n,C.m) - Brt_i * P.ar;

        Frt_i.noalias() = P.Pr * Art_i;
        Fht_i.noalias() = P.Ph * Aht_i;

        // Update equations
        P.Pr.noalias() += C.integration_time_step * (Frt_i + Frt_i.transpose() + P.Qr - P.Pr * Brt_i * P.Pr + P.Ph * Brht_i * P.Ph);
        P.ar.noalias() += C.integration_time_step * ((Art_i - Brt_i * P.Pr).transpose() * P.ar + P.Pr * crt_i + P.Ph * Brht_i * P.ah);
        P.Ph.noalias() += C.integration_time_step * (Fht_i + Fht_i.transpose() + P.Qh - P.Ph * Bht_i * P.Ph);
        P.ah.noalias() += C.integration_time_step * ((Aht_i - Bht_i * P.Ph).transpose() * P.ah + P.Ph * cht_i);
        
        
    }
}

////////////////////////////////////////////////////////////////////////// DG_loop //////////////////////////////////////////////////////////////////////////

void DG_loop(Dg &DG, Dg::Constants &C) {       C.nc++;

    Update_Planning(DG.P,DG.S,DG.C);
    DG_fxn(DG.P,DG.C);

    // Updating Ur, Uh, and xi
    DG.R.Ur = -DG.P.Rr.inverse() * (DG.P.Br.block(0,0,C.n,C.m)).transpose() * (DG.P.Pr * DG.S.xi + DG.P.ar);
    DG.H.Uh = -DG.P.Rh.inverse() * (DG.P.Bh.block(0,0,C.n,C.m)).transpose() * (DG.P.Ph * DG.S.xi + DG.P.ah);
    DG.S.xi +=C.time_step * ((DG.P.A.block(0,0,C.n,C.n)* DG.S.xi)+ ((DG.P.Br.block(0,0,C.n,C.m))* DG.R.Ur)+ ((DG.P.Bh.block(0,0,C.n,C.m)) * DG.H.Uh) + (DG.P.C.block(0,0,C.n,1)));

    // Store them in their matirxes
    DG.S.xi_arr.push_back(DG.S.xi); DG.R.Ur_arr.push_back(DG.R.Ur); DG.H.Uh_arr.push_back(DG.H.Uh);

    //cout<<endl;
}

////////////////////////////////////////////////////////////////////////// Construct_All //////////////////////////////////////////////////////////////////////////

void Construct_All(Dg &DG) {

    Construct_Human(DG.H, DG.C);
    Construct_Robot(DG.R, DG.C);
    Construct_State(DG.S, DG.H, DG.R, DG.C); 
    Construct_Planning(DG.P, DG.S, DG.H, DG.R, DG.C);
}