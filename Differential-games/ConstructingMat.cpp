#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

double I=0.0152,D=0.5,m=1,l=0.1,g=9.81;

vector<string> split(const string &s, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Function to read the CSV file and divide data into vectors
void readCSV(const string &filename,MatrixXd& A,MatrixXd &B, MatrixXd& C) {
    ifstream file(filename);
    string line;

    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }
    MatrixXd mA(2,2), mB(2,1),mC(2,1);
    mB << 0,1/I;
    mA << 0,1,
        0,-D/I;
    mC<<0,0;
    int i=0;
        while (getline(file, line, '|')) {
        vector<string> parts = split(line, '/');
        if (parts.size() == 3) {
            mA(1, 0) = (m * g * l * sin(stod(parts[0])))/I;
            mC(1,0)=-(1/I)*(I*stod(parts[2])+D*stod(parts[1])+m*g*l*cos(stod(parts[0])));

            if (i==100){ cout<< mA<<endl<<mC<<endl;}
            A.block(2*i, 0, 2, 2) = mA;
            B.block(2*i, 0, 2, 1) = mB;
            C.block(2*i, 0, 2, 1) = mC;
            i++;
        }


    }

    file.close();
}

int main() {
    string filename = "sine_points.csv"; // Replace with your CSV file name
    MatrixXd A(2*10000,2),B(2*10000,1),C(2*10000,1);

    readCSV(filename, A, B, C);

    // Print the matrices
    // cout << "Matrix A:" << endl;
     cout << A << endl;

    // cout << "Matrix B:" << endl;
    // cout << B << endl;

    // cout << "Matrix C:" << endl;
    // cout << C << endl;

    return 0;
}
