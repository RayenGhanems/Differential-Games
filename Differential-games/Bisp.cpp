#include <SFML/Graphics.hpp>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <iostream>
#include <cmath>
#include <fstream>

const double T = 60, pps = 0.5;
double offset=M_PI/8,range=M_PI-offset*2,initial_pos=M_PI/2-offset;


// Function to calculate a point on a Catmull-Rom spline
sf::Vector2f catmullRomSpline(const sf::Vector2f& p0, const sf::Vector2f& p1, const sf::Vector2f& p2, const sf::Vector2f& p3, float t) {
    float t2 = t * t;
    float t3 = t2 * t;

    float f0 = -0.5f * t3 + t2 - 0.5f * t;
    float f1 =  1.5f * t3 - 2.5f * t2 + 1.0f;
    float f2 = -1.5f * t3 + 2.0f * t2 + 0.5f * t;
    float f3 =  0.5f * t3 - 0.5f * t2;

    return p0 * f0 + p1 * f1 + p2 * f2 + p3 * f3;
}

// Function to write data to a CSV file
void writeToCSV(const std::string& filename, const std::vector<float>& data) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    for (const auto& value : data) {
        file << value << ",\n";
    }

    file.close();
}

int main() {
    // Seed the random number generator
    srand(static_cast<unsigned int>(time(0)));

    // Create a window
    sf::RenderWindow window(sf::VideoMode(1000, 600), "Catmull-Rom Spline");

    // Create a vector to store the control points
    std::vector<sf::Vector2f> controlPoints;

    // Add the first point at (0, 600)
    controlPoints.push_back(sf::Vector2f(0, 300)); // Adjusted for the radius

    // Loop through x-coordinates from 10 to 100 in increments of 10
    for (float x = 1000/(T/pps); x <= 1000+(T/pps); x += 1000/(T/pps)) {
        // Generate a random y-coordinate between 0 and 10
        int y = rand() % 9;

        // Scale the x and y to fit the window size
        float scaledX = x ; // Scale x to fit the width (800)
        float scaledY = y * 60; // Scale y to fit the height (600)

        // Add the point to the control points vector
        controlPoints.push_back(sf::Vector2f(scaledX, 540 - scaledY - 5)); // Adjusted for the radius
    }

    // Create a vector to store spline points for writing to CSV
    std::vector<float> splinePoints;

    bool write=true;

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

        // Draw the control points
        // for (const auto& point : controlPoints) {
        //     sf::CircleShape circle(5); // Radius of 5
        //     circle.setPosition(point);
        //     circle.setFillColor(sf::Color::Red);
        //     window.draw(circle);
        // }

        // Draw the Catmull-Rom spline
        sf::VertexArray spline(sf::LinesStrip);
        for (int i = 0; i < T/pps; ++i) {
            for (float t = 0.001; t < 1; t += 0.001) {
                sf::Vector2f point = catmullRomSpline(controlPoints[i - 1], controlPoints[i], controlPoints[i + 1], controlPoints[i + 2], t);
                spline.append(sf::Vertex(point, sf::Color::Green));

                // Store spline points for writing to CSV
                if (write && i*1000/(T/pps)<=1000) {
                    splinePoints.push_back((point.y / 600)*(range)-initial_pos);
                }
            }
        }
        write=false;

        window.draw(spline);

        // Display the contents of the window
        window.display();
    }

    // Write spline points to CSV file
    writeToCSV("spline_points.csv", splinePoints);

    return 0;
}