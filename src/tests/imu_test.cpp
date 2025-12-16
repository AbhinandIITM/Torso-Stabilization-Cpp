#include "utils/IMU_server.hpp"
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <thread>
#include <chrono>

namespace beast = boost::beast;
namespace websocket = beast::websocket;
namespace net = boost::asio;
using tcp = net::ip::tcp;

int main() {
    const int IMU_WEBSOCKET_PORT = 8001;
    
    std::cout << "\n========================================\n";
    std::cout << "  IMU WebSocket Server Test\n";
    std::cout << "========================================\n\n";
    
    try {
        net::io_context ioc{1};
        tcp::acceptor acceptor{ioc, {tcp::v4(), IMU_WEBSOCKET_PORT}};
        
        std::cout << "IMU server running on ws://0.0.0.0:" << IMU_WEBSOCKET_PORT << "\n";
        std::cout << "Waiting for Android app connection...\n\n";
        
        // Create IMU tracker
        IMUTracker tracker;  // ✅ Remove utils::

        
        while (true) {
            // Accept connection
            tcp::socket socket{ioc};
            acceptor.accept(socket);
            
            std::cout << "Client connected!\n";
            
            // Upgrade to WebSocket
            websocket::stream<tcp::socket> ws{std::move(socket)};
            ws.accept();
            
            std::cout << "WebSocket connection established.\n";
            std::cout << "Receiving IMU data...\n\n";
            
            int message_count = 0;
            
            // Read messages in a loop
            while (true) {
                try {
                    beast::flat_buffer buffer;
                    ws.read(buffer);
                    
                    std::string message = beast::buffers_to_string(buffer.data());
                    message_count++;
                    
                    // Print raw message every 10 messages
                    if (message_count % 10 == 0) {
                        // std::cout << "\n--- Message #" << message_count << " ---\n";
                        // std::cout << message << "\n";
                    }
                    
                    // Parse JSON
                    try {
                        auto json_data = nlohmann::json::parse(message);
                        
                        // Extract IMU data
                        if (json_data.contains("imu")) {
                            auto& imu = json_data["imu"];
                            
                            // Extract gyroscope
                            Eigen::Vector3d gyro(0, 0, 0);
                            if (imu.contains("gyroscope")) {
                                auto& gyro_vals = imu["gyroscope"]["values"];
                                gyro = Eigen::Vector3d(
                                    gyro_vals[0].get<double>(),
                                    gyro_vals[1].get<double>(),
                                    gyro_vals[2].get<double>()
                                );
                            }
                            
                            // Extract accelerometer
                            Eigen::Vector3d accel(0, 0, 0);
                            if (imu.contains("linear_acceleration")) {
                                auto& accel_vals = imu["linear_acceleration"]["values"];
                                accel = Eigen::Vector3d(
                                    accel_vals[0].get<double>(),
                                    accel_vals[1].get<double>(),
                                    accel_vals[2].get<double>()
                                );
                            }
                            
                            // Get timestamp (dt will be calculated between updates)
                            double dt = 0.001; // Default 100Hz
                            if (json_data.contains("timestamp")) {
                                static double last_timestamp = json_data["timestamp"].get<double>();
                                double current_timestamp = json_data["timestamp"].get<double>();
                                dt = current_timestamp - last_timestamp;
                                std::cout << 1/dt << std::endl;
                                last_timestamp = current_timestamp;
                                
                            }
                            
                            // Update tracker
                            tracker.update(accel, gyro, dt);
                            
                            // Print orientation every 50 messages
                            if (message_count % 5 == 0) {
                                // auto transform = tracker.get_transform();
                                // std::cout << "\n--- Orientation Update #" << message_count << " ---\n";
                                // std::cout << "Gyro:  [" << gyro.transpose() << "] rad/s\n";
                                // std::cout << "Accel: [" << accel.transpose() << "] m/s²\n";
                                // std::cout << "Transform:\n" << transform << "\n\n";
                                
                            }
                        }
                        
                    } catch (const nlohmann::json::exception& e) {
                        std::cerr << "JSON parse error: " << e.what() << "\n";
                    }
                    
                } catch (const beast::system_error& e) {
                    if (e.code() == websocket::error::closed) {
                        std::cout << "\nClient disconnected.\n";
                        std::cout << "Total messages received: " << message_count << "\n\n";
                        break;
                    }
                    throw;
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
