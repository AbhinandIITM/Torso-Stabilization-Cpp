#pragma once

#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <nlohmann/json.hpp>

#include <memory>
#include <mutex>

// Forward declare IMUTracker to avoid including its full header here.
class IMUTracker;

// Type aliases for convenience
using json = nlohmann::json;
typedef websocketpp::server<websocketpp::config::asio> server;

// ---------------- IMUServer -----------------
// Manages a WebSocket server to receive and process IMU data.
class IMUServer {
public:
    // Constructor to set up the server on a specific port.
    IMUServer(int port = 8001);

    // Starts the server's event loop. This is a blocking call.
    void run();

    // Stops the server's event loop gracefully.
    void stop();

    // Thread-safe method to get the latest processed transform data.
    json get_latest_transform() const;

private:
    // The callback function that is executed when a message is received.
    void on_message(websocketpp::connection_hdl hdl, server::message_ptr msg);

    int port;
    server server_;
    std::shared_ptr<IMUTracker> imu_tracker;

    // Thread-safe storage for the latest received data.
    // The mutex is mutable to allow locking in const methods.
    mutable std::mutex mutex_;
    json latest_data;
    json latest_tf;
};

