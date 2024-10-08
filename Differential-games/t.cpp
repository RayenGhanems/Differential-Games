#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>




using namespace std;
using namespace chrono;


int t=0;

void Plus(int& o){
    duration<double> pause_duration(0.6);
    this_thread::sleep_for(pause_duration);
    cout<<"Plus: "<<++o<<endl;
}

void Min(){
    while(t<2){
    }
    while(1){
        duration<double> pause_duration(1);
        this_thread::sleep_for(pause_duration);
        cout<<"Min : "<<t<<endl;
    }
}

int main(){
    
    auto startt=high_resolution_clock::now();
    std::thread t2(Min);
    while(t<5){
        Plus(t);
    }
    t2.detach();


    auto endt=high_resolution_clock::now();
    auto duration = duration_cast<microseconds>((endt - startt));
    cout <<endl<< "Time taken: " << duration.count() << " microseconds" << endl;

    return 0;
}