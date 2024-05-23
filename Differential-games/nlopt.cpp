#include <iomanip>
#include <iostream>
#include <vector>

#include <nlopt.hpp>

// Define your objective function here
double myfunc(const std::vector<double>& x, std::vector<double>& grad, void* my_func_data) {
  // Implement your objective function logic here
  // This example calculates the square root of the second variable
  if (!grad.empty()) {
    grad[0] = 0.0; // Gradient for first variable (can be set to 0 if not used)
    grad[1] = 0.5 / sqrt(x[1]); // Gradient for second variable
  }
  return sqrt(x[1]);
}

// Define your constraint function here
struct my_constraint_data {
  double a;
  double b;
};

double myconstraint(const std::vector<double>& x, std::vector<double>& grad, void* data) {
  // Implement your constraint logic here
  // This example defines a constraint based on data structure
  my_constraint_data* d = reinterpret_cast<my_constraint_data*>(data);
  double a = d->a;
  double b = d->b;
  if (!grad.empty()) {
    grad[0] = 3 * a * pow(a * x[0] + b, 2); // Gradient for first variable
    grad[1] = -1.0; // Gradient for second variable
  }
  return pow(a * x[0] + b, 3) - x[1]; // Constraint function based on data
}

int main() {
  // Create NLopt object and set parameters
  nlopt::opt opt(nlopt::LD_MMA, 2);
  std::vector<double> lb(2);
  lb[0] = -HUGE_VAL;
  lb[1] = 0;
  opt.set_lower_bounds(lb);
  opt.set_min_objective(myfunc, nullptr); // Set user-defined objective function

  // Define constraint data
  my_constraint_data data[2] = {{2, 0}, {-1, 1}};

  // Add inequality constraints with data pointers
  opt.add_inequality_constraint(myconstraint, &data[0], 1e-8);
  opt.add_inequality_constraint(myconstraint, &data[1], 1e-8);

  opt.set_xtol_rel(1e-4); // Set tolerance for solution

  // Initial guess for variables
  std::vector<double> x(2);
  x[0] = 1.234;
  x[1] = 5.678;
  double minf;

  try {
    nlopt::result result = opt.optimize(x, minf);
    std::cout << "found minimum at f(" << x[0] << "," << x[1] << ") = "
              << std::setprecision(10) << minf << std::endl;
  } catch (std::exception& e) {
    std::cout << "nlopt failed: " << e.what() << std::endl;
  }

  return 0;
}


/////// To Run  g++ nlopt.cpp -o out -lnlopt -lm