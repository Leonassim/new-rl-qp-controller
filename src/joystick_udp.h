#pragma once

#include <mc_rtc/logging.h>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstring>

/** Gamepad state over UDP, sent by scripts/joystick/rlqp_joy.py (the pad sits on another PC: rhps1t, or tardis through
 *  an ssh relay). Fixed 32-byte packets, little endian:
 *    "RLJ1" | u32 seq | u8 n_axes | u8 n_buttons | u16 pad | i16 axes[8] | u16 buttons | u16 pad
 *  Raw Linux joystick values (/dev/input/js*), converted here exactly as mc_joystick_plugin does. */
struct JoystickUdp
{
  static constexpr size_t kPacket = 32;
  static constexpr int kAxes = 8;

  ~JoystickUdp() { close(); }

  bool open(int port)
  {
    close();
    fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if(fd_ < 0) { return false; }
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port = htons(static_cast<uint16_t>(port));
    if(::bind(fd_, reinterpret_cast<sockaddr *>(&a), sizeof(a)) != 0 || ::fcntl(fd_, F_SETFL, O_NONBLOCK) != 0)
    {
      mc_rtc::log::error("[JoystickUdp] cannot bind UDP port {}: {}", port, std::strerror(errno));
      close();
      return false;
    }
    port_ = port;
    return true;
  }

  void close()
  {
    if(fd_ >= 0) { ::close(fd_); }
    fd_ = -1;
  }

  bool isOpen() const { return fd_ >= 0; }

  /** Drains the socket, keeps the newest valid packet. Cheap: a few non-blocking recv per tick. */
  void poll()
  {
    if(fd_ < 0) { return; }
    unsigned char buf[64];
    for(int k = 0; k < 64; ++k)
    {
      const ssize_t n = ::recv(fd_, buf, sizeof(buf), 0);
      if(n < 0) { break; }
      if(n != static_cast<ssize_t>(kPacket) || std::memcmp(buf, "RLJ1", 4) != 0) { continue; }
      uint32_t seq;
      std::memcpy(&seq, buf + 4, 4);
      // Out-of-order or replayed packets (UDP) are ignored, a sender restart (seq back near 0) is accepted.
      if(received_ && seq <= seq_ && seq_ - seq < 1000) { continue; }
      seq_ = seq;
      std::memcpy(axes_, buf + 12, sizeof(axes_));
      std::memcpy(&buttons_, buf + 28, 2);
      last_ = std::chrono::steady_clock::now();
      received_ = true;
    }
  }

  /** Seconds since the last valid packet, 1e9 before the first. */
  double age() const
  {
    if(!received_) { return 1e9; }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - last_).count();
  }

  bool everReceived() const { return received_; }
  uint32_t seq() const { return seq_; }
  uint16_t buttons() const { return buttons_; }

  /** mc_joystick_plugin's scale: -((v + MIN) / (MAX - MIN)), 0.5 at rest, 1 for a stick pushed up or left. */
  double axis(int i) const
  {
    if(i < 0 || i >= kAxes) { return 0.5; }
    return -((static_cast<double>(axes_[i]) - 32768.0) / 65535.0);
  }

  int port() const { return port_; }

private:
  int fd_ = -1;
  int port_ = 0;
  bool received_ = false;
  uint32_t seq_ = 0;
  int16_t axes_[kAxes] = {0, 0, 0, 0, 0, 0, 0, 0};
  uint16_t buttons_ = 0;
  std::chrono::steady_clock::time_point last_;
};
