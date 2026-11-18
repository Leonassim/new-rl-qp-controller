#pragma once
// Shared memory between NewRLQPController and the vision process (scripts/vision/rhps1_vision.py).
//   /rhps1_vision_pose  controller -> vision, ring of poses at the control rate (created by the controller)
//   /rhps1_vision_map   vision -> controller, latest height map (created by the vision process)
// Times are the controller's own clock (seconds since it started), so simulation runs keep their time base.
// Poses are in the controller's estimated world frame (observer pipeline, MCWaiko on the robot); the map is
// built in that same frame, so only drift between a frame and its use reaches the policy.
// Byte layout is fixed (offsets asserted below); rhps1_vision.py mirrors it.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace vision_link
{

constexpr uint32_t MAGIC = 0x31535652;  // "RVS1"
constexpr int POSE_RING = 64;
constexpr int MAP_N = 48;
constexpr const char * POSE_SHM = "/rhps1_vision_pose";
constexpr const char * MAP_SHM = "/rhps1_vision_map";

struct Pose
{
  double t;               // controller time, s
  double pelvis[7];       // estimated: x y z qw qx qy qz, body to world
  double head[7];         // estimated HEAD_P_LINK pose, same convention
  double pelvis_true[7];  // simulation ground truth; NaN on the robot
  double soles[2][4];     // L, R sole bottom centre: x y z yaw (estimated)
  double loaded[2];       // 1 when the sole carries more than 100 N
};
static_assert(sizeof(Pose) == 256, "Pose layout");

struct PoseBlock
{
  uint32_t magic;
  uint32_t ring;
  uint64_t count;  // poses written; the newest is poses[(count - 1) % ring]
  Pose poses[POSE_RING];
};
static_assert(offsetof(PoseBlock, poses) == 16, "PoseBlock layout");

struct MapBlock
{
  uint32_t magic;
  uint32_t n;
  uint64_t seq;      // seqlock: odd while the writer is inside
  double t_frame;    // controller time of the newest depth frame in the map
  double res;        // m
  double origin[2];  // world x, y of the lower corner of cell (0, 0)
  float h[MAP_N * MAP_N];  // terrain height, cell (ix, iy) at ix * n + iy; NaN = never seen
};
static_assert(offsetof(MapBlock, h) == 48, "MapBlock layout");

template<typename T>
T * mapShm(const char * name, bool create)
{
  int fd = create ? shm_open(name, O_RDWR | O_CREAT, 0666) : shm_open(name, O_RDONLY, 0);
  if(fd < 0 && create)
  {
    // A segment left by another user: fs.protected_regular refuses O_CREAT on it in the sticky /dev/shm, even to
    // root (the controller runs as root under hrpsys). Replace it; the vision process reopens the new one.
    shm_unlink(name);
    fd = shm_open(name, O_RDWR | O_CREAT, 0666);
  }
  if(fd < 0) { return nullptr; }
  if(create) { fchmod(fd, 0666); }  // readable and writable by the vision process whatever the umask
  if(create && ftruncate(fd, sizeof(T)) != 0)
  {
    close(fd);
    return nullptr;
  }
  struct stat st;
  if(fstat(fd, &st) != 0 || st.st_size < static_cast<off_t>(sizeof(T)))
  {
    close(fd);
    return nullptr;
  }
  void * p = mmap(nullptr, sizeof(T), create ? PROT_READ | PROT_WRITE : PROT_READ, MAP_SHARED, fd, 0);
  close(fd);
  return p == MAP_FAILED ? nullptr : static_cast<T *>(p);
}

/** Controller side: publishes poses, reads maps without ever waiting. */
class Link
{
public:
  Link()
  {
    pose_ = mapShm<PoseBlock>(POSE_SHM, true);
    if(pose_)
    {
      pose_->magic = MAGIC;
      pose_->ring = POSE_RING;
      pose_->count = 0;
    }
    std::fill(std::begin(map_.h), std::end(map_.h), NAN);
  }
  ~Link()
  {
    if(pose_) { munmap(pose_, sizeof(PoseBlock)); }
    if(mapIn_) { munmap(const_cast<MapBlock *>(mapIn_), sizeof(MapBlock)); }
  }
  Link(const Link &) = delete;
  Link & operator=(const Link &) = delete;

  bool ok() const { return pose_ != nullptr; }

  void publish(const Pose & p)
  {
    if(!pose_) { return; }
    const uint64_t c = pose_->count;
    pose_->poses[c % POSE_RING] = p;
    std::atomic_thread_fence(std::memory_order_release);
    pose_->count = c + 1;
  }

  /** Copies the latest consistent map into map(); false while none has arrived. */
  bool poll()
  {
    if(!mapIn_)
    {
      if(++retry_ % 200 != 1) { return have_; }
      mapIn_ = mapShm<MapBlock>(MAP_SHM, false);
      if(!mapIn_) { return have_; }
    }
    const volatile MapBlock * m = mapIn_;
    const uint64_t s1 = m->seq;
    if((s1 & 1u) || s1 == lastSeq_ || m->magic != MAGIC || m->n != MAP_N) { return have_; }
    std::atomic_thread_fence(std::memory_order_acquire);
    std::memcpy(&tmp_, const_cast<const MapBlock *>(mapIn_), sizeof(MapBlock));
    std::atomic_thread_fence(std::memory_order_acquire);
    if(m->seq != s1) { return have_; }
    map_ = tmp_;
    lastSeq_ = s1;
    have_ = true;
    return true;
  }

  /** Forget the mapping (the vision process may have restarted on a new segment); the next poll reopens. */
  void dropMap()
  {
    if(mapIn_) { munmap(const_cast<MapBlock *>(mapIn_), sizeof(MapBlock)); }
    mapIn_ = nullptr;
    retry_ = 0;
  }

  const MapBlock & map() const { return map_; }
  bool have() const { return have_; }

private:
  PoseBlock * pose_ = nullptr;
  const MapBlock * mapIn_ = nullptr;
  MapBlock map_{}, tmp_{};
  uint64_t lastSeq_ = 0;
  unsigned retry_ = 0;
  bool have_ = false;
};

} // namespace vision_link
