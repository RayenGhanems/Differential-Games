#include <armadillo>
#include <iostream>
#include <chrono>
#include <math.h>

#include "nlopt.h"

using namespace arma;
using namespace std;
using namespace chrono;


void Dg(cube a, cube b, cube c, cube d);
void Estimation(cube a, cube b, cube c, cube d, colvec ξo);
double objective_function(const double* x, double* grad, void* data);

// Define initial values
mat phf = {{0, 0}, {0, 0}};
colvec ahf = {0, 0};
mat prf = phf;
colvec arf = {0, 0};
mat Qr = {{10, 0}, {0, 0.1}};
mat Qh = {{20, 0}, {0, 0.1}};
mat mA = {{0, 1}, {-0.1, -0.1}};
colvec mBr = {0, 0.1};
colvec mBh = {0, 0.1};
colvec mC = {0, 0.1};

// Assuming Rrt, Rrh, and Rh are identity matrices
mat Rr = {1};
mat Rh = {1};
mat Rrh = {1};
const int DelP = 1000, DelE = 5;
double T = 0.001;

colvec Ur(1), Uh(1), ξ = {1, 1}, error = {0};

// Declare array to store past Uh values
colvec Uh_arr[DelE];
colvec ξ_arr[DelE];

// Create cubes with all slices the same as the original matrices
cube A(2, 2, 15 * DelP);
cube Br(2, 1, 15 * DelP);
cube Bh(2, 1, 15 * DelP);
cube C(2, 1, 15 * DelP);

mat Brt_i, Bht_i, Brht_i, Art_i, Aht_i, crt_i, cht_i, Frt_i, Fht_i;

// Pre-allocate storage for loop iterations (now matrices)
mat Pr(2, 2);
mat Ph(2, 2);
colvec ar(2);
colvec ah(2);


int main() {
  for (int i = 0; i < 15 * DelP; i++) {
    A.slice(i) = mA;
    Br.slice(i) = mBr;
    Bh.slice(i) = mBh;
    C.slice(i) = mC;
  }
  int a, b;

  // Start clock
  auto start = chrono::high_resolution_clock::now();

  for (int i = 0; i < 15; i++) {
    a = i * DelP;
    b = a + DelP - 1;
    Dg(A(span::all, span::all, span(a, b)), Br(span::all, span::all, span(a, b)), Bh(span::all, span::all, span(a, b)), C(span::all, span::all, span(a, b)));

    // Updating Ur, Uh and ξ
    Ur = -inv(Rr) * Br.slice(b).t() * (Pr * ξ + ar);
    Uh = -inv(Rh) * Bh.slice(b).t() * (Ph * ξ + ah);
    ξ += T * (a * ξ + Br.slice(b) * Ur + Bh.slice(b) * Uh + C.slice(b));

    // Store Uh in the array
    Uh_arr[i % DelE] = Uh;
    ξ_arr[i % DelE] = ξ;

    if (i >= DelE) {
      a -= DelE;
      Estimation(A(span::all, span::all, span(a, b)), Br(span::all, span::all, span(a, b)), Bh(span::all, span::all, span(a, b)), C(span::all, span::all, span(a, b)), ξ_arr[(i + 1) % DelE]);
    }
  }

  // End clock
  auto end = chrono::high_resolution_clock::now();

  auto duration = chrono::duration_cast<  chrono::microseconds>(end - start);

  // Print
  cout << "Pr:\n" << Pr << endl;
  cout << "ar:\n" << ar << endl;
  cout << "Ph:\n" << Ph << endl;
  cout << "ah:\n" << ah << endl;

  cout << "Ur:\n" << Ur << endl;
  cout << "Uh:\n" << Uh << endl;
  cout << "ξ:\n" << ξ << endl;

  cout << "Time taken: " << duration.count() << " microseconds" << endl; // Cout Time
}


void Dg(cube mA, cube mBr, cube mBh, cube mC) {
  // Initialize loop variables with actual values
  Pr = prf;
  Ph = phf;
  ar = arf;
  ah = ahf;

  // Implement the loop
  for (int i = 0; i < DelP; ++i) {
    mat a = mA.slice(i), br = mBr.slice(i), bh = mBh.slice(i), c = mC.slice(i);

    Brt_i = br * inv(Rr) * br.t();
    Bht_i = bh * inv(Rh) * bh.t();
    Brht_i = br * inv(Rr) * Rrh * inv(Rr) * br.t();

    Art_i = a - Bht_i * Ph;
    Aht_i = a - Brt_i * Pr;
    crt_i = c - Bht_i * ah;
    cht_i = c - Brt_i * ar;

    Frt_i = Pr * Art_i;
    Fht_i = Ph * Aht_i;

    // Update equations
    Pr = Pr + T * (Frt_i + Frt_i.t() + Qr - Pr * Brt_i * Pr + Ph * Brht_i * Ph);
    ar = ar + T * ((Art_i - Brt_i * Pr).t() * ar + Pr * crt_i + Ph * Brht_i * ah);
    Ph = Ph + T * (Fht_i + Fht_i.t() + Qh - Ph * Bht_i * Ph);
    ah = ah + T * ((Aht_i - Bht_i * Ph).t() * ah + Ph * cht_i);
  }
}


double objective_function(const double* x, double* grad, void* data) {
    colvec* error_ptr = static_cast<colvec*>(data);
    double sum = 0;
    for (int i = 0; i < error_ptr->n_elem; i++) {
    sum += abs(error_ptr->at(i));
    }
    return sum;
}

void Estimation(cube a, cube b, cube c, cube d, colvec ξo) {
  int s;
  ξ = ξo;

  for (int i = 0; i < DelE; i++) {
    s = i + DelP;
    Dg(a(span::all, span::all, span(i, s)), b(span::all, span::all, span(i, s)), c(span::all, span::all, span(i, s)), d(span::all, span::all, span(i, s)));

    // ** NLopt Integration **
    const int n_params = 2; // Assuming you want to optimize ar(0) and ah(0)
    double lower_bounds[n_params] = {0, 0};
    double upper_bounds[n_params] = {HUGE_VAL,HUGE_VAL};

    // Perform NLopt optimization (assuming you want to optimize ar(0) and ah(0))
    nlopt_opt opt = nlopt_create(NLOPT_LN_BOBYQA, n_params);
    nlopt_set_lower_bounds(opt, lower_bounds);
    nlopt_set_upper_bounds(opt, upper_bounds);
    nlopt_set_maxeval(opt, 6); // Use the specified maxeval from your parameters

    // ** Replace with your desired initial guess for the parameters to optimize **
    double x[n_params] = {a(0, 0, i), c(0, 0, i)};

    double min_f;
    nlopt_result result = nlopt_optimize(opt, x, &min_f);

    // Check for optimization success
    if (result != NLOPT_SUCCESS) {
      cout << "NLopt optimization for step " << i << " failed!" << endl;
    } else {
      // Update parameters with optimized values
      a(0, 0, i) = x[0];
      c(0, 0, i) = x[1];
    }

    nlopt_destroy(opt); // Clean up NLopt object after each optimization

    Uh = -inv(Rh) * c.slice(s).t() * (Ph * ξ + c(0, 0, i));
    ξ += T * (i * ξ + b.slice(s) * Ur + c.slice(s) * Uh + d.slice(s));

    // Calculate error using Uh_arr[i] (assuming element-wise absolute difference)
    error += abs(Uh_arr[i % DelE] - Uh);
  }

  cout << error << "\n";
}
