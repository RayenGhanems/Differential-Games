#include <armadillo>
#include <iostream>
#include <chrono>

#include "nlopt.h"

using namespace arma;
using namespace std;
using namespace chrono;


void Dg(mat a,mat b,mat c,mat d);
void Estimation(mat a,mat b,mat c,mat d,colvec ξo);
void Define();

const int dim=2,DelP = 100, DelE = 250, Tf=1000, n=Tf+DelP+1;
const double T=0.001;
int j,b;


// Define initial values
mat phf(2, 2), prf(2, 2), Qr(2, 2), Qh(2, 2), mA(2, 2), Brt_i, Bht_i, Brht_i, Art_i, Aht_i, crt_i, cht_i, Frt_i, Rr(1, 1), Rh(1, 1), Rrh(1, 1), Fht_i, Pr(2, 2), Ph(2, 2), a, br, bh, c;
vec ahf(2), arf(2), mBr(2), mBh(2), mC(2), Ur(1), Uh(1), ξ0(2), ξ(2), error(2), Uh_arr[DelE], ξ_arr[DelE], ar(2), ah(2);
mat A(dim * n, dim), Br(dim * n, 1), Bh(dim * n, 1), C(dim * n, 1);




int main() {
  Define();
  for(int i=0;i<n;i++){
    j=2*i;
    A( arma::span(j,j+1),arma::span::all) =  mA;
    Br(arma::span(j,j+1),arma::span::all) = mBr;
    Bh(arma::span(j,j+1),arma::span::all) = mBh;
    C (arma::span(j,j+1),arma::span::all) =  mC;
  }

  // Start clock
  auto start = high_resolution_clock::now();

  for(int i=0;i<Tf;i++){   j=2*i;b=(j+DelP*2)-1;
    Dg(A(arma::span(j, b), arma::span::all), Br(arma::span(j, b), arma::span::all), Bh(arma::span(j, b), arma::span::all), C(arma::span(j, b), arma::span::all));j=2*i;b=(j+DelP*2)-1;

    // // Updating Ur, Uh and ξ
    // Ur=-inv(Rr)*Br(arma::span(j, j+1), arma::span::all).t()*(Pr*ξ+ar);
    // Uh=-inv(Rh)*Bh(arma::span(j, j+1), arma::span::all).t()*(Ph*ξ+ah);
    // ξ+=T*(A(arma::span(j, j+1), arma::span::all)*ξ+Br(arma::span(j, j+1), arma::span::all)*Ur+Bh(arma::span(j, j+1), arma::span::all)*Uh+C(arma::span(j, j+1), arma::span::all));

    // // Store Uh and ξ in the array
    // Uh_arr[i % DelE] = Uh; 
    // ξ0=ξ_arr[i % DelE];
    // ξ_arr[i % DelE] = ξ;

    //if(i>DelE){           j-=DelE/2-2;
      //Estimation(A(arma::span(j, b), arma::span::all), Br(arma::span(j, b), arma::span::all), Bh(arma::span(j, b), arma::span::all), C(arma::span(j, b), arma::span::all), ξ0);
    //}
        
  }
  

  // End clock
  auto end = high_resolution_clock::now();
  
  
  auto duration = duration_cast<microseconds>((end - start)/Tf);
  
    // Print 
  cout << "Pr:\n" << Pr << endl;
  cout << "ar:\n" << ar << endl;
  cout << "Ph:\n" << Ph << endl;
  cout << "ah:\n" << ah << endl;

  cout << "Ur:\n" << Ur << endl;
  cout << "Uh:\n" << Uh << endl;
  cout << "ξ:\n" << ξ << endl;
    
  cout << "Time taken: " << duration.count() << " microseconds" << endl;    // Print Time


  return 0;
}


void Dg(mat dgA,mat dgBr,mat dgBh,mat dgC){
  // Initialize loop variables with actual values
  Pr = prf;
  Ph = phf;
  ar = arf;
  ah = ahf;
    
  // Implement the loop
  for (int i = 0; i < DelP; ++i) {    j =2*i;
    a=dgA(arma::span(j,j+1),arma::span::all) ;br=dgBr(arma::span(j,j+1),arma::span::all) ;bh=dgBh(arma::span(j,j+1),arma::span::all) ;c=dgC(arma::span(j,j+1),arma::span::all) ;

    Brt_i = br * inv(Rr) * br.t();
    Bht_i = bh * inv(Rh) * bh.t();
    Brht_i = br* inv(Rr) * Rrh * inv(Rr) * br.t();

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

void Estimation(mat eA,mat eBr,mat eBh ,mat eC, colvec ξo){

  ξ=ξo;
  error = {0};
  for(int i=0;i<DelE;i++){    j=2*i;b=(j+DelP*2)-1;
    Dg(eA(arma::span(j, b), arma::span::all),eBr(arma::span(j, b), arma::span::all),eBh(arma::span(j, b), arma::span::all),eC(arma::span(j, b), arma::span::all));
    Uh=-inv(Rh)*Bh(arma::span(j, j+1), arma::span::all).t()*(Ph*ξ+ah);
    Ur=-inv(Rr)*Br(arma::span(j, j+1), arma::span::all).t()*(Pr*ξ+ar);
    ξ+=T*(A(arma::span(j, j+1), arma::span::all)*ξ+Br(arma::span(j, j+1), arma::span::all)*Ur+Bh(arma::span(j, j+1), arma::span::all)*Uh+C(arma::span(j, j+1), arma::span::all));

   
    error += abs(Uh_arr[i % DelE] - Uh);  
  }

  cout<<error ;
}

void Define(){

    phf = {{0, 0},
           {0, 0}};
    prf = phf;
    Qr = {{10, 0},
          {0, 0.1}};
    Qh = {{20, 0},
          {0, 0.1}};
    mA = {{0, 1},
          {-0.1, -0.1}};

    ahf = {0, 0};
    arf = {0, 0};
    mBr = {0, 0.1};
    mBh = {0, 0.1};
    mC = {0, 0.1};
    Ur = {0};
    Uh = {0};
    ξ0 = {0, 0};
    ξ = {1, 1};
    error = {0, 0};

    Rr = {1};
    Rh = {1};
    Rrh = {1};
}