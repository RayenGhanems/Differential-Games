#ifndef DG_STRUCTURE_H
#define DG_STRUCTURE_H

#include <vector>
#include <iostream>
#include <string>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

struct Dg {
    struct Constants {
        const int m = 1;
        const int n = 2 * m; 
        const double time_step = 0.001;
        const double task_duration = 10;
        double current_time = 0;    
        const double integration_time_step = 0.01;
        const double planning_horizon = 1;
        const double estimation_horizon = 0.2;
        const int integration_step = integration_time_step / time_step;
        int ne = estimation_horizon / time_step;
        int np = planning_horizon / time_step;
        int nc = -1;
        bool loop=true;
        int ntotal = task_duration / time_step;
        vector<double> x;
    };

    struct Robot {
        const double I, m, l, D;
        MatrixXd qr, rr, rrh;
        VectorXd Ur;
        vector<VectorXd> Ur_arr;
        Robot() : I(0.0152), m(1), l(0.1), D(0.5), qr(2, 2), rr(1, 1), rrh(1, 1), Ur(1) {}
    };

    struct Human {
        const double I, m, l, D;
        MatrixXd qh, rh, rhr;
        VectorXd Uh;
        vector<VectorXd> Uh_arr;
        Human() : I(0.0497), m(0.427), l(0.3411), D(0), qh(2, 2), rh(1, 1), rhr(1, 1), Uh(1) {}
    };

    struct State {
        MatrixXd A, B, C, Xd;
        VectorXd xi;
        vector<VectorXd> xi_arr;
        double I, D, M, l, g = 9.81;
    };

    struct Planning {
        MatrixXd A, Br, Bh, C, Qr, Rr, Rrh, Pr, Qh, Rh, Ph;
        VectorXd ar, ah;
    };

    struct Estimation {
        MatrixXd A, Br, Bh, C, Qr, Qh, Rr, Rh, Rrh, Ph, Pr;
        VectorXd ar, ah, xi, Uh, Ur;
        vector<VectorXd> Uh_arr;
        vector<vector<float>> X;
    };

    Constants C;
    Human H;
    Robot R;
    State S;
    Planning P;
    Estimation E;

};

void Construct_Human(Dg::Human& H, const Dg::Constants &C);
void Construct_Robot(Dg::Robot& R, const Dg::Constants &C);
vector<string> split(const string& s, char delimiter, const Dg::Constants &C);
void readCSV(const string& filename, Dg::State& S, const Dg::Constants &C);
void Construct_State(Dg::State& a, Dg::Human& H, Dg::Robot& R, const Dg::Constants &C);
void Construct_Planning(Dg::Planning& P, const Dg::State& S, Dg::Human& H, Dg::Robot& R, const Dg::Constants &C);
void Construct_Estimation(Dg::Estimation& E, const Dg::State& S, Dg::Human& H, Dg::Robot& R, const Dg::Constants &C);
void Update_Planning(Dg::Planning& P, const Dg::State& S, const Dg::Constants &C);
void DG_fxn(Dg::Planning& P, const Dg::Constants &C);
void DG_loop(Dg& DG, Dg::Constants &C);
void Construct_All(Dg& DG);


#endif