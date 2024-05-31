#include <SFML/Graphics.hpp>
#include <cmath>
#include <iostream>
#include <fstream>

const double T = 60, pps = 0.5;
double offset = M_PI / 8, range = M_PI - offset * 2, initial_pos = M_PI / 2 - offset;

// Function to write data to a CSV file
void writeToCSV(const std::string &filename, const std::vector<float> &pos,const std::vector<float> &vel,const std::vector<float> &acc) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    for (int i=0;i<10000;i++) {
        if(i==0||i==10000-1){
            file << pos[i]<<"/0/0|\n";
        }
        else {
            if(i==1||i==10000-2){
                file << pos[i]<<"/"<<vel[i-1]<<"/0|\n";
            }
            else {file << pos[i]<<"/"<<vel[i-1]<<"/"<<acc[i-2] << "|\n";}
        }
    }

    file.close();
}

int main() {
    // Create a window
    sf::RenderWindow window(sf::VideoMode(1000, 600), "Sine Function");

    // Create a vector to store spline points for writing to CSV
    std::vector<float> sinePoints;
    sinePoints.push_back(0.0);
    std::vector<float> sineVelocity;
    sineVelocity.push_back(0.0);
    std::vector<float> sineAcceleration;
    sineAcceleration.push_back(0.0);

    bool write = true,done =true;

    // Main loop
    while (window.isOpen()) {
        // Process events
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        // Clear the window
        window.clear(sf::Color::Black);

        // Draw the sine function
        sf::VertexArray sine(sf::LinesStrip);
        for (float t = 0; t <= 110; t += 0.01) {

            float x = t*10; 
            float y = std::sin(t); 
            y = (y + 1) * 300;
            sine.append(sf::Vertex(sf::Vector2f(x, y), sf::Color::Green));

            // Store spline points for writing to CSV
            if (write && x <= 1100) {
                float pos=sinePoints.back();
                sinePoints.push_back((y / 600) * range - initial_pos);
                if(x>1&&x<1100){
                    float vel=sineVelocity.back();
                    sineVelocity.push_back((sinePoints.back()-pos)/T);
                    if(x>2&&x<1100-1){
                        sineAcceleration.push_back((sineVelocity.back()-vel)/T);
                    }
                }
            }
            else{
                if(done){
                    // Write spline points to CSV file
                    writeToCSV("sine_points.csv", sinePoints,sineVelocity,sineAcceleration);
                    done=false;
                }
            }
            
        }
        write = false;

        window.draw(sine);

        // Display the contents of the window
        window.display();
    }

    

    return 0;
}
