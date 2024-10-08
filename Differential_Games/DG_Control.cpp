#include "DG_Structure.h"

#include <vector>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <chrono>
#include <thread>
#include <nlopt.hpp>
#include <iomanip>
#include <atomic>
#include <algorithm>

using namespace std;
using namespace Eigen;
using namespace chrono;
using namespace nlopt;

void Update_Estimation();
double Estimation_fxn(const vector<double>& x, vector<double>& grad, void* data);
void Estimation_Loop(Dg &DG);
void DG_Loop(Dg &DG);
void writeToCSV(const vector<VectorXd>& Ur_arr,const vector<VectorXd>& Uh_arr);
void writeX(const vector<vector<float>>& data);

Dg DG;
int enc;
float TT=500.5;

int main() {
    
    Construct_All(DG);

    //////////////////////////////////////////////////////////////////////////  Start Threads  //////////////////////////////////////////////////////////////////////////
    thread Estimation_thread(Estimation_Loop, ref(DG));
    thread DG_thread(DG_Loop, ref(DG));

    //////////////////////////////////////////////////////////////////////////  Main Loop  //////////////////////////////////////////////////////////////////////////////////

    auto start = high_resolution_clock::now();

    DG_thread.join();

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    DG.C.loop = false;

    Estimation_thread.join();

    // writeToCSV(DG.R.Ur_arr, DG.H.Uh_arr);
    // writeX(DG.E.X);

    std::cout << "\nTime taken: " << duration.count() << " microseconds         DG" << endl; 
    cout<< "DG Time: "<< TT<<endl;

    return 0;
}


////////////////////////////////////////////////////////////////////////// ESTIMATION functions //////////////////////////////////////////////////////////////////////////

void Update_Estimation() {   
    int in = enc - DG.C.ne;
    DG.E.A = DG.S.A.block(0, in * DG.C.n,DG.C.n, (DG.C.ne + DG.C.np) * DG.C.n);
    DG.E.Br = DG.S.B.block(0, in * DG.C.m,DG.C.n, (DG.C.ne + DG.C.np) * DG.C.m);
    DG.E.Bh = DG.S.B.block(0, in * DG.C.m,DG.C.n, (DG.C.ne + DG.C.np) * DG.C.m);
    DG.E.C = DG.S.C.block(0, in,DG.C.n, (DG.C.ne + DG.C.np));

    DG.E.xi = DG.S.xi_arr[in];
    DG.E.Uh_arr = DG.H.Uh_arr;

    DG.E.ah.setZero(); DG.E.ar.setZero(); DG.E.Ph.setZero(); DG.E.Pr.setZero();
}

double Estimation_fxn(const vector<double>& x, vector<double>& grad, void* data) {
    MatrixXd Brt_i, Brht_i, Bht_i, cht_i, crt_i, Qh(DG.C.n,DG.C.n);
    MatrixXd Aht_i, Art_i, Frt_i, Fht_i;
    MatrixXd Rr_inv = DG.E.Rr.inverse(), Rh_inv = DG.E.Rh.inverse();
    VectorXd error_vectors(DG.C.m);  error_vectors.setZero();
    double error = 0;
    int i;
    Qh << x[0], 0, 0, x[1];

    for (int i = 0; i < DG.C.ne; i++) {
        DG.E.ah.setZero(); DG.E.ar.setZero(); DG.E.Ph.setZero(); DG.E.Pr.setZero();
        for (int j = DG.C.np - 1; j > 0; j -= DG.C.integration_step) {  
            Brt_i.noalias() = DG.E.Br.block(0, (j + i),DG.C.n,DG.C.m) * Rr_inv * (DG.E.Br.block(0, (j + i),DG.C.n,DG.C.m)).transpose();
            Bht_i.noalias() = DG.E.Bh.block(0, (j + i),DG.C.n,DG.C.m) * Rh_inv * (DG.E.Bh.block(0, (j + i),DG.C.n,DG.C.m)).transpose();
            Brht_i.noalias() = DG.E.Br.block(0, (j + i),DG.C.n,DG.C.m) * Rr_inv * DG.E.Rrh * Rr_inv * (DG.E.Br.block(0, (j + i),DG.C.n,DG.C.m)).transpose();
            Art_i.noalias() = DG.E.A.block(0, (j + i) * DG.C.n,DG.C.n,DG.C.n) - Bht_i * DG.E.Ph;
            Aht_i.noalias() = DG.E.A.block(0, (j + i) * DG.C.n,DG.C.n,DG.C.n) - Brt_i * DG.E.Pr;
            crt_i.noalias() = DG.E.C.block(0, (j + i),DG.C.n,DG.C.m) - Bht_i * DG.E.ah;
            cht_i.noalias() = DG.E.C.block(0, (j + i),DG.C.n,DG.C.m) - Brt_i * DG.E.ar;

            Frt_i.noalias() = DG.E.Pr * Art_i;
            Fht_i.noalias() = DG.E.Ph * Aht_i;

            // Update equations
            DG.E.Pr.noalias() += DG.C.integration_time_step * (Frt_i + Frt_i.transpose() + DG.E.Qr - DG.E.Pr * Brt_i * DG.E.Pr + DG.E.Ph * Brht_i * DG.E.Ph);
            DG.E.ar.noalias() += DG.C.integration_time_step * ((Art_i - Brt_i * DG.E.Pr).transpose() * DG.E.ar + DG.E.Pr * crt_i + DG.E.Ph * Brht_i * DG.E.ah);
            DG.E.Ph.noalias() += DG.C.integration_time_step * (Fht_i + Fht_i.transpose()  + Qh - DG.E.Ph * Bht_i * DG.E.Ph);
            DG.E.ah.noalias() += DG.C.integration_time_step * ((Aht_i - Bht_i * DG.E.Ph).transpose() * DG.E.ah + DG.E.Ph * cht_i);
        }

        DG.E.Uh = -DG.E.Rh.inverse() * DG.E.Bh.block(0, i,DG.C.n,DG.C.m).transpose() * (DG.E.Ph * DG.E.xi + DG.E.ah);
        DG.E.Ur = -DG.E.Rr.inverse() * DG.E.Br.block(0, i,DG.C.n,DG.C.m).transpose() * (DG.E.Pr * DG.E.xi + DG.E.ar);
        DG.E.xi += DG.C.time_step * (DG.E.A.block(0, i * DG.C.n,DG.C.n,DG.C.n) * DG.E.xi + DG.E.Br.block(0, i,DG.C.n,DG.C.m) * DG.E.Ur + DG.E.Bh.block(0, i,DG.C.n,DG.C.m) * DG.E.Uh + DG.E.C.block(0, i,DG.C.n, 1));
        error_vectors += (DG.E.Uh - DG.E.Uh_arr[enc - DG.C.ne + i]) * (DG.E.Uh - DG.E.Uh_arr[enc - DG.C.ne + i]);
    }
    
    error = error_vectors.sum();
    return error;
}

void Estimation_Loop(Dg &DG) {
    Construct_Estimation(DG.E, DG.S, DG.H, DG.R,DG.C);
    vector<float> v;
    double minf, i = 0;
    vector<double> lb, ub;
    opt opt(LN_BOBYQA, 2);
    vector<double> x;
    x.push_back(10);     x.push_back(0.001);
    //x.push_back(50);    x.push_back(0.1);
    lb.push_back(0);    lb.push_back(0);
    ub.push_back(1000); ub.push_back(1000);

    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);
    opt.set_maxeval(8);
    //opt0.set_default_initial_step({50,0.01});
    opt.set_initial_step({ 10, 0.001 });
    opt.set_min_objective(Estimation_fxn, nullptr);

    while (DG.C.loop) {
        if (DG.C.current_time <= DG.C.estimation_horizon) {  
            this_thread::sleep_for(microseconds(100));
        }
        else {          
            enc = DG.C.nc; 
            i++;

            Update_Estimation();

            opt.optimize(x, minf);

            if (minf < 0.001) {
                opt.set_initial_step({ minf, minf });
            }
            if(isnan(minf)){
                minf=0;
            }
            std::cout << DG.C.current_time << " :   found minimum at f(" << x[0] << "," << x[1] << ") =  " << minf << "  " << i << endl;
            v.push_back(DG.C.current_time);
            v.push_back(x[0]);
            v.push_back(x[1]);
            DG.E.X.push_back(v);
            v.clear();        
        }
    }
}

void DG_Loop(Dg& DG) {

    high_resolution_clock loopClock; high_resolution_clock::time_point timer;

    for (DG.C.current_time = 0; DG.C.current_time < DG.C.task_duration; DG.C.current_time += DG.C.time_step) {
        timer = high_resolution_clock::now();

        DG_loop(DG,DG.C);

        if(TT > duration_cast<microseconds>(high_resolution_clock::now() - timer).count()){ TT = duration_cast<microseconds>(high_resolution_clock::now() - timer).count();}
        // while ((loopClock.now() - timer).count() < 1e6);
    }
}

void writeToCSV(const vector<VectorXd>& Ur_arr,const vector<VectorXd>& Uh_arr) {
    ofstream file("/home/ray/Desktop/a./New Folder/outputs_DG/Ur_Uh_DG.csv");
    if (file.is_open()) {

        for (size_t i = 0; i < Ur_arr.size(); ++i) {

            const VectorXd& vec_r = Ur_arr[i];
            const VectorXd& vec_h = Uh_arr[i];

            if (vec_r.size() != vec_h.size()) {
                cerr << "Error: Vectors in Ur_arr and Uh_arr at index " << i << " have different sizes." << endl;
                return;
            }

            for (int j = 0; j < vec_r.size(); ++j) {
                file << vec_r(j);
                file << "/"; 
                file << vec_h(j);
            }

            file << "\n";
        }

        // Close the file after writing
        file.close();
        std::cout <<endl<< "Data successfully written " << endl;
    } else {
        cerr << "Unable to open file " <<  endl;
    }
}

void writeX(const vector<vector<float>>& data) {
    std::ofstream file("/home/ray/Desktop/a./New Folder/outputs_DG/X.csv");

    if (file.is_open()) {
        // Write column headers (optional)
        file << "time,x1,x2\n";

        // Write the data
        for (const auto& row : data) {
            if (row.size() == 3) {
                file << row[0] << "," << row[1] << "," << row[2] << "\n";
            }
        }

        file.close();
        std::cout << "Data successfully written to file" << "\n";
    } else {
        std::cerr << "Could not open file " << "\n";
    }
}