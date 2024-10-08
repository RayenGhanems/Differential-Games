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

    for (int i=1;i<100000;i++) {
        file << pos[i]<<"/"<<vel[i]<<"/"<<acc[i] << "|\n";}
        
    

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
        for (float t = 0; t <= 100; t += 0.001) {
            
            float x = t*100; 
            float y = std::sin(t); 
            float v = std::cos(t);
            float a = -(y);
            sine.append(sf::Vertex(sf::Vector2f(x, (y+initial_pos)*600/range), sf::Color::Green));

            // Store spline points for writing to CSV
            if (write && x <= 100/0.001) {
                sinePoints.push_back(y);
                sineVelocity.push_back(v);
                sineAcceleration.push_back(a);
                
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
